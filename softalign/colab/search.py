import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict, namedtuple
from functools import lru_cache
import csv
import time

import Score_align as score_



def pad_enc(t, target_len):
    pad_len = target_len - t.shape[0]
    if pad_len > 0:
        return jnp.pad(t, ((0, pad_len), (0, 0)))
    else:
        return t

def pad_aux(x, target_len):
    pad_len = target_len - x.shape[0]
    if pad_len > 0:
        return jnp.pad(x, ((0, pad_len), (0, 0), (0, 0)))
    else:
        return x

def get_bucketed_length(length, thresholds=[100, 200, 300, 400, 600, 800, 1000, 1420]):
    for t in thresholds:
        if length <= t:
            return t
    return length

def bucket_items(items_with_info, thresholds):
    buckets = defaultdict(list)
    for item in items_with_info:
        length = item[3]
        bucket = get_bucketed_length(length, thresholds)
        buckets[bucket].append(item)
    return buckets

# --- Batch size adaptive based on query AND target length ---
def get_batch_size_for_lengths_softmax(query_len, target_len):
    """
    Batch size calculator for Softmax. This version uses a tiered approach 
    to be aggressive for small inputs while remaining safe for large ones.
    """
    total_elements = query_len * target_len
    if total_elements <= 50000:  # e.g., 500x100
        return 128
    
    if total_elements <= 250000: # e.g., 500x500
        return 64
        
    # For large inputs, scale down based on a memory budget.
    # This constant represents the max elements (batch_size * L_q * L_t)
    # that can fit in memory for a large batch. Tune for your GPU.
    SAFE_MAX_ELEMENTS_PER_BATCH = 4_000_000 
    
    if query_len == 0 or target_len == 0:
        return 128

    calculated_batch_size = SAFE_MAX_ELEMENTS_PER_BATCH // total_elements
    
    if calculated_batch_size < 1:
        return 1
    return min(calculated_batch_size, 32)


def get_batch_size_for_lengths_sw(query_len, target_len):
    """
    Batch size calculator for Smith-Waterman, which is more memory intensive.
    Uses smaller, more conservative batch sizes than the softmax version.
    """
    total_elements = query_len * target_len

    if total_elements <= 10000:  
        return 128*2
    
    if total_elements <= 25000:
        return 64*2
    

        
    # Use a more conservative memory budget for SW
    SAFE_MAX_ELEMENTS_PER_BATCH = 2_000_000 # Half of softmax budget
    
    if query_len == 0 or target_len == 0:
        return 64

    calculated_batch_size = SAFE_MAX_ELEMENTS_PER_BATCH // total_elements
    if calculated_batch_size < 1:
        return 1

    return min(calculated_batch_size, 16)

# --- Cached and jit-compiled score function generator ---
@lru_cache(maxsize=None)
def get_score_fn(bucket_q, bucket_t, model_type):
    @jax.jit
    def score_fn(enc1_batch, enc2_batch, X_q_batch, X_t_batch, lengths):
        if model_type == "Smith-Waterman":
            aln = score_.align_SW(enc1_batch, enc2_batch, lengths, 1e-7, 0., 0.)
        elif model_type == "Softmax":
            aln = score_.align_argmax(enc1_batch, enc2_batch, lengths, 1e-7)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        col_mask_q = jnp.arange(bucket_q)[None, :] >= lengths[:, 0:1]
        mask1 = ~col_mask_q[:, None, :] * ~col_mask_q[:, :, None]
        col_mask_t = jnp.arange(bucket_t)[None, :] >= lengths[:, 1:2]
        mask2 = ~col_mask_t[:, None, :] * ~col_mask_t[:, :, None]
        return score_.get_LDDTloss(
            X_q_batch[:, :, 1], X_t_batch[:, :, 1], aln, mask1, mask2, 1e-4,
            values_ANG=[0.5, 1, 2, 4]
        )
    return score_fn

def process_bucket_group(query_enc, X_query, enc2_all, X_t_all, lengths_all, bucket_t, model_type, true_l_query):
    scores_device = []
    # Select the appropriate batch size calculator based on the model type
    if model_type == "Smith-Waterman":
        batch_size = get_batch_size_for_lengths_sw(query_enc.shape[0], bucket_t)
    else: # Default to softmax for "Softmax" 
        batch_size = get_batch_size_for_lengths_softmax(query_enc.shape[0], bucket_t)
    
    score_fn = get_score_fn(query_enc.shape[0], bucket_t, model_type)
    enc1_full = jnp.broadcast_to(query_enc[None, :, :], (batch_size, *query_enc.shape))
    X_q_full = jnp.broadcast_to(X_query[None, :, :, :], (batch_size, *X_query.shape))
    n = enc2_all.shape[0]
    for i in range(0, n, batch_size):
        current_batch_size = min(batch_size, n - i)
        enc1 = enc1_full[:current_batch_size]
        X_q = X_q_full[:current_batch_size]
        enc2 = jax.lax.dynamic_slice_in_dim(enc2_all, i, current_batch_size, axis=0)
        X_t = jax.lax.dynamic_slice_in_dim(X_t_all, i, current_batch_size, axis=0)
        lengths = jax.lax.dynamic_slice_in_dim(lengths_all, i, current_batch_size, axis=0)
        scores_batch = score_fn(enc1, enc2, X_q, X_t, lengths)
        scores_device.append(scores_batch)
    return scores_device

def one_vs_all_optimized(query_enc, X_query, preprocessed_buckets, model_type="Softmax", true_l_query=None):
    all_scores_device, all_ids = [], []
    if true_l_query is None:
        true_l_query = query_enc.shape[0]
    for bucket_t, (enc2_all, X_t_all, ids_all, lengths_t) in sorted(preprocessed_buckets.items()):
        #print(f"Dispatching scoring for targets <= {bucket_t} length ({len(ids_all)} items)")
        lengths_all = jnp.stack([jnp.full_like(lengths_t, true_l_query), lengths_t], axis=1)
        bucket_scores_device = process_bucket_group(
            query_enc, X_query, enc2_all, X_t_all, lengths_all, bucket_t, model_type, true_l_query)
        all_scores_device.extend(bucket_scores_device)
        all_ids.extend(ids_all)
    #print("Waiting for all scoring computations to finish...")
    if all_scores_device:
        all_scores_device[-1].block_until_ready()
    all_scores_cpu = [np.array(s) for s in all_scores_device]
    final_scores_flat = np.concatenate(all_scores_cpu)
    id_to_score_map = dict(zip(all_ids, final_scores_flat))
    original_order_ids = []
    for bucket_t, (_, _, ids_all, _) in sorted(preprocessed_buckets.items()):
        original_order_ids.extend(ids_all)
    final_scores = [id_to_score_map[id] for id in original_order_ids]
    return final_scores, original_order_ids

#A container for the reusable, pre-processed target data.
TargetData = namedtuple('TargetData', ['preprocessed_buckets', 'items_with_info'])

# Function to perform the expensive, one-time setup.
def setup_target_data(dicti_encodings, dicti_inputs,thresholds = [100, 150, 200, 300, 400, 600, 800, 1000, 1420]
):
    print("--- Starting One-Time Target Setup ---")
    start_time = time.time()
    items_with_info = []
    for k, v_enc in dicti_encodings.items():
        v_input = dicti_inputs[k]
        items_with_info.append((k, v_enc, v_input[0], v_enc.shape[0]))
    items_with_info.sort(key=lambda x: x[3])
    
    buckets = bucket_items(items_with_info, thresholds)

    print("Dispatching pre-processing for all buckets...")
    preprocessed_buckets = {}
    last_op = None
    for bucket_t, group in buckets.items():
        if not group: continue
        enc2_all = jnp.stack([pad_enc(item[1], bucket_t) for item in group])
        X_t_all = jnp.stack([pad_aux(item[2][0], bucket_t) for item in group])
        ids_all = [item[0] for item in group]
        lengths_t = jnp.array([item[3] for item in group], dtype=jnp.int32)
        preprocessed_buckets[bucket_t] = (enc2_all, X_t_all, ids_all, lengths_t)
        last_op = lengths_t

    if last_op is not None:
        print("Waiting for data to be moved to device...")
        last_op.block_until_ready()
    
    end_time = time.time()
    print(f"--- One-Time Setup Finished in {end_time - start_time:.2f} seconds ---")
    return TargetData(preprocessed_buckets, items_with_info)


def compute_scores_for_query(query_id, target_data, model_type="Softmax", l_query_pad=None):
    print(f"\nProcessing query: {query_id}")
    start_time = time.time()
    
    preprocessed_buckets = target_data.preprocessed_buckets
    items_with_info = target_data.items_with_info
    
    all_sorted_ids = [item[0] for item in items_with_info]
    try:
        query_index = all_sorted_ids.index(query_id)
    except ValueError:
        raise ValueError(f"Query ID '{query_id}' not found in pre-processed data.")
    _, query_enc_raw, X_query_raw, l_query = items_with_info[query_index]

    if l_query_pad is not None:
        if l_query_pad < l_query:
            raise ValueError(f"Pad length {l_query_pad} < query length {l_query}")
        query_enc = pad_enc(query_enc_raw, l_query_pad)
        X_query = pad_aux(X_query_raw[0], l_query_pad)
    else:
        query_enc = query_enc_raw
        X_query = X_query_raw

    scores, sorted_ids = one_vs_all_optimized(
        query_enc=query_enc,
        X_query=X_query,
        preprocessed_buckets=preprocessed_buckets,
        model_type=model_type,
        true_l_query=l_query
    )

    id_score_pairs = list(zip(sorted_ids, scores))
    sorted_by_score = sorted(id_score_pairs, key=lambda x: x[1], reverse=True)
    csv_filename = f"scores_sorted_{query_id}.csv"
    output_path = "./output/" + csv_filename

    import os; os.makedirs("./output", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score"])
        writer.writerows(sorted_by_score)
    
    end_time = time.time()
    print(f"✅ Saved scores for {query_id} to `{csv_filename}` in {end_time - start_time:.2f} seconds.")

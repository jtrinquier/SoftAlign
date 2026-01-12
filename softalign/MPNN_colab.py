# /content/SoftAlign/softalign/MPNN.py
# Rewritten to:
# - remove SafeKey (no mutable RNG state inside modules)
# - use Haiku RNG correctly via hk.next_rng_key()
# - avoid deprecated jax.tree_map usage (removed with SafeKey)
# - silence int64 warnings by using int32 for indices
# - fix GELU usage (no double "approximate" arg)

import functools
import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp


# Exact GELU (approximate=False) as a callable with one argument
Gelu = functools.partial(jax.nn.gelu, approximate=False)


def gather_edges(edges, neighbor_idx):
    """Features [B,N,N,C] at neighbor indices [B,N,K] => [B,N,K,C]."""
    neighbors = jnp.tile(jnp.expand_dims(neighbor_idx, -1), [1, 1, 1, edges.shape[-1]])
    edge_features = jnp.take_along_axis(edges, neighbors, axis=2)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """Features [B,N,C] at neighbor indices [B,N,K] => [B,N,K,C]."""
    neighbors_flat = neighbor_idx.reshape([neighbor_idx.shape[0], -1])  # [B, NK]
    neighbors_flat = jnp.tile(jnp.expand_dims(neighbors_flat, -1), [1, 1, nodes.shape[2]])  # [B, NK, C]
    neighbor_features = jnp.take_along_axis(nodes, neighbors_flat, axis=1)  # [B, NK, C]
    neighbor_features = neighbor_features.reshape(list(neighbor_idx.shape[:3]) + [-1])  # [B, N, K, C]
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    """Features [B,N,C] at neighbor indices [B,K] => [B,K,C]."""
    idx_flat = jnp.tile(jnp.expand_dims(neighbor_idx, -1), [1, 1, nodes.shape[2]])
    neighbor_features = jnp.take_along_axis(nodes, idx_flat, axis=1)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = jnp.concatenate([h_neighbors, h_nodes], axis=-1)
    return h_nn


class PositionalEncodings(hk.Module):
    def __init__(self, num_embeddings, max_relative_feature=32, name=None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = hk.Linear(num_embeddings, name="embedding_linear")

    def __call__(self, offset, mask):
        # Ensure int32 indices for JAX default x64-disabled setups
        offset = jax.lax.convert_element_type(offset, jnp.int32)
        mask = jax.lax.convert_element_type(mask, jnp.int32)

        d = (
            jnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        )
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 2)
        E = self.linear(jax.lax.convert_element_type(d_onehot, jnp.float32))
        return E


class ProteinFeatures(hk.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        num_chain_embeddings=16,  # kept for compatibility even if unused here
        name=None,
    ):
        super().__init__(name=name)
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25  # node_in kept for compatibility
        self.edge_embedding = hk.Linear(edge_features, with_bias=False, name="edge_embedding")
        self.norm_edges = hk.LayerNorm(-1, create_scale=True, create_offset=True, name="norm_edges")

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = jnp.expand_dims(mask, 1) * jnp.expand_dims(mask, 2)
        dX = jnp.expand_dims(X, 1) - jnp.expand_dims(X, 2)
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=3) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max

        # approx_min_k expects k <= dimension
        k = np.minimum(self.top_k, X.shape[1])
        D_neighbors, E_idx = jax.lax.approx_min_k(D_adjust, k, reduction_dimension=-1)
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count).reshape([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = jnp.expand_dims(D, -1)
        RBF = jnp.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, X, mask, residue_idx, chain_labels):
        # Optional coordinate noise augmentation (correct RNG usage)
        if self.augment_eps > 0:
            key = hk.next_rng_key()
            X = X + self.augment_eps * jax.random.normal(key, X.shape)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = jnp.cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = [
            self._rbf(D_neighbors),            # Ca-Ca
            self._get_rbf(N, N, E_idx),        # N-N
            self._get_rbf(C, C, E_idx),        # C-C
            self._get_rbf(O, O, E_idx),        # O-O
            self._get_rbf(Cb, Cb, E_idx),      # Cb-Cb
            self._get_rbf(Ca, N, E_idx),       # Ca-N
            self._get_rbf(Ca, C, E_idx),       # Ca-C
            self._get_rbf(Ca, O, E_idx),       # Ca-O
            self._get_rbf(Ca, Cb, E_idx),      # Ca-Cb
            self._get_rbf(N, C, E_idx),        # N-C
            self._get_rbf(N, O, E_idx),        # N-O
            self._get_rbf(N, Cb, E_idx),       # N-Cb
            self._get_rbf(Cb, C, E_idx),       # Cb-C
            self._get_rbf(Cb, O, E_idx),       # Cb-O
            self._get_rbf(O, C, E_idx),        # O-C
            self._get_rbf(N, Ca, E_idx),       # N-Ca
            self._get_rbf(C, Ca, E_idx),       # C-Ca
            self._get_rbf(O, Ca, E_idx),       # O-Ca
            self._get_rbf(Cb, Ca, E_idx),      # Cb-Ca
            self._get_rbf(C, N, E_idx),        # C-N
            self._get_rbf(O, N, E_idx),        # O-N
            self._get_rbf(Cb, N, E_idx),       # Cb-N
            self._get_rbf(C, Cb, E_idx),       # C-Cb
            self._get_rbf(O, Cb, E_idx),       # O-Cb
            self._get_rbf(C, O, E_idx),        # C-O
        ]
        RBF_all = jnp.concatenate(tuple(RBF_all), axis=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        d_chains = jax.lax.convert_element_type(d_chains, jnp.int32)
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]

        E_positional = self.embeddings(
            jax.lax.convert_element_type(offset, jnp.int32),
            E_chains,
        )

        E = jnp.concatenate((E_positional, RBF_all), axis=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class PositionWiseFeedForward(hk.Module):
    def __init__(self, num_hidden, num_ff, name=None):
        super().__init__()
        self.W_in = hk.Linear(num_ff, with_bias=True, name=(name or "") + "_W_in")
        self.W_out = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W_out")
        self.act = Gelu

    def __call__(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class dropout_cust(hk.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def __call__(self, x):
        if self.rate == 0.0:
            return x
        return hk.dropout(hk.next_rng_key(), self.rate, x)

class EncLayer(hk.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        name=None,
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = dropout_cust(dropout)
        self.dropout2 = dropout_cust(dropout)
        self.dropout3 = dropout_cust(dropout)


        self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=(name or "") + "_norm1")
        self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=(name or "") + "_norm2")
        self.norm3 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=(name or "") + "_norm3")

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W1")
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W2")
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W3")

        self.W11 = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W11")
        self.W12 = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W12")
        self.W13 = hk.Linear(num_hidden, with_bias=True, name=(name or "") + "_W13")

        self.act = Gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, name=(name or "") + "_dense")

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer."""

        # Node update
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, 1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message

        dh = jnp.sum(h_message, axis=-2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V

        # Edge update
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, 1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)

        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class ENC:
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers=1,
        k_neighbors=64,
        augment_eps=0.05,
        dropout=0.1,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
        )

        self.W_e = hk.Linear(hidden_dim, with_bias=True, name="W_e")
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, name=f"enc{i}")
            for i in range(num_encoder_layers)
        ]


    def __call__(self, X, mask, residue_idx, chain_encoding_all):
        """Graph-conditioned encoder."""
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)

        # Initialize node states
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)

        mask_attend = gather_nodes(jnp.expand_dims(mask, -1), E_idx).squeeze(-1)
        mask_attend = jnp.expand_dims(mask, -1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V

import numpy as np 
import jax 
import jax.numpy as jnp 


def pad_(X_1, mask_1, res_1, chain_1, X_2, mask_2, res_2, chain_2, max_len):
    # Pad each element in X_1 and X_2 to the same length
    X1_padded = np.array([np.pad(X, ((0, max_len - X.shape[0]), (0, 0), (0, 0)), 'constant') for X in X_1])
    X2_padded = np.array([np.pad(X, ((0, max_len - X.shape[0]), (0, 0), (0, 0)), 'constant') for X in X_2])

    # Create masks, residue numbers, and chain IDs for X1_padded and X2_padded
    mask1_padded = np.array([np.pad(mask, (0, max_len - mask.shape[0]), 'constant') for mask in mask_1])
    res1_padded = np.array([np.pad(res, (0, max_len - res.shape[0]), 'constant', constant_values=-100) for res in res_1])
    chain1_padded = np.array([np.pad(chain, (0, max_len - chain.shape[0]), 'constant') for chain in chain_1])

    mask2_padded = np.array([np.pad(mask, (0, max_len - mask.shape[0]), 'constant') for mask in mask_2])
    res2_padded = np.array([np.pad(res, (0, max_len - res.shape[0]), 'constant', constant_values=-100) for res in res_2])
    chain2_padded = np.array([np.pad(chain, (0, max_len - chain.shape[0]), 'constant') for chain in chain_2])


    # Get the lengths of each element in X_1 and X_2
    len_X1 = np.array([X.shape[0] for X in X_1])
    len_X2 = np.array([X.shape[0] for X in X_2])

    # Combine the padded arrays, masks, residue numbers, and chain IDs into a single return tuple
    return X1_padded, mask1_padded, res1_padded, chain1_padded, X2_padded, mask2_padded, res2_padded, chain2_padded, np.column_stack((len_X1, len_X2))
def pad_tmalign(X_1, mask_1, res_1, chain_1, X_2, mask_2, res_2, chain_2, tmal, max_len):
    # Pad each element in X_1 and X_2 to the same length
    X1_padded = np.array([np.pad(X, ((0, max_len - X.shape[0]), (0, 0), (0, 0)), 'constant') for X in X_1])
    X2_padded = np.array([np.pad(X, ((0, max_len - X.shape[0]), (0, 0), (0, 0)), 'constant') for X in X_2])

    # Create masks, residue numbers, and chain IDs for X1_padded and X2_padded
    mask1_padded = np.array([np.pad(mask, (0, max_len - mask.shape[0]), 'constant') for mask in mask_1])
    res1_padded = np.array([np.pad(res, (0, max_len - res.shape[0]), 'constant', constant_values=-100) for res in res_1])
    chain1_padded = np.array([np.pad(chain, (0, max_len - chain.shape[0]), 'constant') for chain in chain_1])

    mask2_padded = np.array([np.pad(mask, (0, max_len - mask.shape[0]), 'constant') for mask in mask_2])
    res2_padded = np.array([np.pad(res, (0, max_len - res.shape[0]), 'constant', constant_values=-100) for res in res_2])
    chain2_padded = np.array([np.pad(chain, (0, max_len - chain.shape[0]), 'constant') for chain in chain_2])

    # Create TMALN
    TMALN = np.ones((len(X_1), max_len)) * -1
    for i in range(len(X_1)):
        tail1 = X_1[i].shape[0]
        TMALN[i, :tail1] = tmal[i][:tail1]

    # Get the lengths of each element in X_1 and X_2
    len_X1 = np.array([X.shape[0] for X in X_1])
    len_X2 = np.array([X.shape[0] for X in X_2])

    # Combine the padded arrays, masks, residue numbers, and chain IDs into a single return tuple
    return X1_padded, mask1_padded, res1_padded, chain1_padded, X2_padded, mask2_padded, res2_padded, chain2_padded, TMALN, np.column_stack((len_X1, len_X2))



def create_test_train(id1,id2,X_1,X_2,chain_1,chain_2,mask_1,mask_2,res_1,res_2,nb_test = 50,tma = True,tmaln = None):

  np.random.seed(0)
  print(len(X_1))
  p = np.random.permutation(len(X_1))

  X_1 = X_1[p]
  X_2 = X_2[p]
  chain_1 = chain_1[p]
  chain_2 = chain_2[p]
  res_1 = res_1[p]
  res_2 = res_2[p]
  mask_1 = mask_1[p]
  mask_2 = mask_2[p]
  if tma:
    tmaln = tmaln[p]

  id1 = id1[p]
  id2 = id2[p]

  X1test = X_1[-nb_test:] ; X2test = X_2[-nb_test:] ; chain1test = chain_1[-nb_test:] ; chain2test = chain_2[-nb_test:]; res1test = res_1[-nb_test:]; res2test = res_2[-nb_test:]; mask1test = mask_1[-nb_test:]; mask2test = mask_2[-nb_test:]
  id1test = id1[-nb_test:] ; id2test = id2[-nb_test:]
  if tma:
    tmalntest = tmaln[-nb_test:]
  k = []
  for l in range(len(X_1)-nb_test):
    if id1[l] not in id1test and id2[l] not in id2test:
      k.append(l)

  k = np.array(k)
  print("nb of training pairs",len(k))
  dicti = {}
  if tma:
    dicti["train"] = (X_1[k],X_2[k],chain_1[k],chain_2[k],mask_1[k],mask_2[k],res_1[k],res_2[k],tmaln[k])
    dicti["test"] = (X1test,X2test,chain1test,chain2test,mask1test,mask2test,res1test,res2test,tmalntest)
  else:
    dicti["train"] = (X_1[k],X_2[k],chain_1[k],chain_2[k],mask_1[k],mask_2[k],res_1[k],res_2[k])
    dicti["test"] = (X1test,X2test,chain1test,chain2test,mask1test,mask2test,res1test,res2test)
  return dicti
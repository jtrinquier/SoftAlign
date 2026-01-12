import numpy as np
import jax.numpy as jnp
import jax

def _np_len_pw(x, use_jax=True):
  '''compute pairwise distance'''
  _np = jnp if use_jax else np

  x_norm = _np.square(x).sum(-1)
  xx = _np.einsum("...ia,...ja->...ij",x,x)
  sq_dist = x_norm[...,:,None] + x_norm[...,None,:] - 2 * xx

  # due to precision errors the values can sometimes be negative
  if use_jax: sq_dist = jax.nn.relu(sq_dist)
  else: sq_dist[sq_dist < 0] = 0

  # return euclidean pairwise distance matrix
  return _np.sqrt(sq_dist + 1e-8)


def get_LDDTloss_old(x1,x2,aln,lens,mask1,mask2,t2, max1 = 300,values_ANG = [0.5,1,2,4],use_jax=True):
    """
    mask1 and mask2 coming from the padding
    t2 temperature in the sigmoid

    """
  #Compute distance matrices
    DM1 = _np_len_pw(x1,use_jax)*mask1
    DM2 = _np_len_pw(x2,use_jax)*mask2

    DM1 = DM1.at[:].set(DM1*(1-jnp.eye(max1)[None,:]))# remove diagonal

    DM1 = jnp.array(DM1)
    DM2 = jnp.array(DM2)
    DM2_al = aln@DM2@jnp.transpose(aln,axes = [0,2,1])
    DM2_al = DM2_al.at[:].set(DM2_al*(1-jnp.eye(max1)[None,:]))

    first_mask = (DM1 == 0) | (DM1>15) #| (DM2_al>15)
    second_mask = (DM2_al < 10**-4) | (DM2_al>15)#not aligned positions
    mask_tot = (first_mask) | (second_mask)

    diff = jnp.sqrt((DM1-DM2_al)**2+10**-8)

    diff_new = diff*(1-mask_tot) + 10000*mask_tot
    temp1 = jnp.sum(1-jax.nn.sigmoid((diff_new[:,:,:,None]-np.array(values_ANG)[None,None,None,:])*t2**-1),axis = (1,2))
    temp2 = (np.sum(1-first_mask,axis = (1,2))[:,None])
    frac = temp1/temp2
    return np.mean(frac,axis = 1)#(frac1+frac2+frac3+frac4)/4

import jax.numpy as jnp
import numpy as np
import jax

def get_LDDTloss(x1, x2, aln, lens,mask1, mask2, t2, max1=300, values_ANG=[0.5, 1, 2, 4], use_jax=True):
    """
    Compute the LDDT loss between two distance matrices.

    Parameters:
    - x1, x2: Input coordinate tensors.
    - aln: Alignment matrix.
    - mask1, mask2: Padding masks.
    - t2: Temperature for sigmoid scaling.
    - max1: Maximum sequence length.
    - values_ANG: Distance thresholds.
    - use_jax: Whether to use JAX computations.

    Returns:
    - LDDT loss averaged over the batch.
    """
    # Compute pairwise distance matrices
    DM1 = _np_len_pw(x1, use_jax) * mask1
    DM2 = _np_len_pw(x2, use_jax) * mask2

    # Remove diagonal values
    DM1 *= (1 - jnp.eye(max1)[None, :])

    # Apply alignment transformation
    DM2_al = aln @ DM2 @ jnp.transpose(aln, axes=[0, 2, 1])
    DM2_al *= (1 - jnp.eye(max1)[None, :])

    # Mask invalid regions
    first_mask = (DM1 == 0) | (DM1 > 15)
    second_mask = (DM2_al < 1e-4) | (DM2_al > 15)
    mask_tot = first_mask | second_mask

    # Compute absolute differences with numerical stability
    diff = jnp.sqrt((DM1 - DM2_al) ** 2 + 1e-8)
    diff = jnp.where(mask_tot, 1e4, diff)  # Assign large value to masked entries

    # Compute loss fraction
    values_ANG = jnp.array(values_ANG)
    temp1 = jnp.sum(1 - jax.nn.sigmoid((diff[..., None] - values_ANG[None, None, None, :]) / t2), axis=(1, 2))
    temp2 = jnp.sum(~first_mask, axis=(1, 2))[:, None]

    return jnp.mean(temp1 / temp2, axis=1)

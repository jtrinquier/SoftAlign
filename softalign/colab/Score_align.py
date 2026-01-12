import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import numpy as np

import SW as sw

#### Alignment functions


@jax.jit
def sim_matrix_(encodings_padded1, encodings_padded2):
    return jnp.einsum("nij,nkj->nik",encodings_padded1,encodings_padded2)

def soft_max_single(sim_matrix, lens, t):
    """
    Softmax on a single sim_matrix
    """
    max_len_1, max_len_2 = sim_matrix.shape

    mask_1 = jnp.arange(max_len_1) < lens[0]
    mask_2 = jnp.arange(max_len_2) < lens[1]

    mask = mask_1[:, None] * mask_2[None, :]
    masked_sim_matrix = jnp.where(mask, sim_matrix, -100000)

    soft_aln = jnp.sqrt(10**-9+
        jax.nn.softmax(t**-1*masked_sim_matrix, axis=-1) *
        jax.nn.softmax(t**-1*masked_sim_matrix, axis=-2)
    )
    return  jnp.where(mask, soft_aln, 0)

def argmax_single(sim_matrix, lens, t):
    """
    Argmax on a single sim_matrix
    """
    max_len_1, max_len_2 = sim_matrix.shape

    mask_1 = jnp.arange(max_len_1) < lens[0]
    mask_2 = jnp.arange(max_len_2) < lens[1]

    mask = mask_1[:, None] * mask_2[None, :]
    masked_sim_matrix = jnp.where(mask, sim_matrix, -100000)

    # Argmax along the specified axes
    argmax_aln_1 = jnp.argmax(masked_sim_matrix, axis=-1)
    argmax_aln_2 = jnp.argmax(masked_sim_matrix, axis=-2)

    print(argmax_aln_2.shape,argmax_aln_1.shape)

    # Create one-hot vectors from argmax indices
    one_hot_aln_1 = jax.nn.one_hot(argmax_aln_1, num_classes=max_len_2)
    one_hot_aln_2 = jax.nn.one_hot(argmax_aln_2, num_classes=max_len_1).T
    print(one_hot_aln_2.shape,one_hot_aln_1.shape)
    argmax_aln = one_hot_aln_1*one_hot_aln_2

    return jnp.where(mask, argmax_aln, 0)

my_sw_func = jax.jit(sw.sw_affine(batch=True))

@jax.jit
def align(encodings_padded1, encodings_padded2,lens,t):
    sim_matrix = sim_matrix_(encodings_padded1, encodings_padded2)
    return  vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix, lens,t)


@jax.jit
def align_argmax(encodings_padded1, encodings_padded2,lens,t):
    sim_matrix = sim_matrix_(encodings_padded1, encodings_padded2)
    return  vmap(argmax_single, in_axes=(0, 0, None))(sim_matrix, lens,t)

@jax.jit
def align_SW(encodings_padded1, encodings_padded2,lens,t,open_ = -3.,gap_ =-1.):
    sim_matrix = sim_matrix_(encodings_padded1, encodings_padded2)
    scores,soft_aln  = my_sw_func(sim_matrix, lens, open_,gap_,t)
    return soft_aln


##### Score functions


def _np_len_pw(x):
  '''compute pairwise distance'''
  _np = jnp

  x_norm = _np.square(x).sum(-1)
  xx = _np.einsum("...ia,...ja->...ij",x,x)
  sq_dist = x_norm[...,:,None] + x_norm[...,None,:] - 2 * xx

  # due to precision errors the values can sometimes be negative
  sq_dist = jax.nn.relu(sq_dist)
  #else: sq_dist[sq_dist < 0] = 0

  # return euclidean pairwise distance matrix
  return _np.sqrt(sq_dist + 1e-8)

@jax.jit
def get_LDDTloss(x1,x2,aln,mask1,mask2,t2,values_ANG = [0.5,1,2,4]):

    """
    mask1 and mask2 coming from the padding
    t2 temperature in the sigmoid
    """
  #Compute distance matrices
    max1 = x1.shape[1]
    DM1 = _np_len_pw(x1)*mask1
    DM2 = _np_len_pw(x2)*mask2

    DM1 = DM1.at[:].set(DM1*(1-jnp.eye(max1)[None,:]))# remove diagonal

    DM2_al = aln@DM2@jnp.transpose(aln,axes = [0,2,1])
    DM2_al = DM2_al.at[:].set(DM2_al*(1-jnp.eye(max1)[None,:]))

    first_mask = (DM1 == 0) | (DM1>15) #| (DM2_al>15)
    second_mask = (DM2_al < 10**-4) | (DM2_al>15)#not aligned positions
    mask_tot = (first_mask) | (second_mask)

    diff = jnp.sqrt((DM1-DM2_al)**2+10**-8)

    diff_new = diff*(1-mask_tot) + 10000*mask_tot
    temp1 = jnp.sum(1-jax.nn.sigmoid((diff_new[:,:,:,None]-jnp.array(values_ANG)[None,None,None,:])*t2**-1),axis = (1,2))
    temp2 = (jnp.sum(1-first_mask,axis = (1,2))[:,None])
    frac = temp1/temp2
    return jnp.mean(frac,axis = 1)#(frac1+frac2+frac3+frac4)/4


@jax.jit
def sum_ali(aln):
  return jnp.sum(aln,axis = (-1,-2))

@jax.jit
def max_ali(aln):
  return jnp.sum(jnp.max(aln,axis = -1),axis = -1)
# Code from the Smooth-Smith-Waterman paper
# Written by Sergey Ovchinnikov and Sam Petti
# Spring 2021
import jax
import jax.numpy as jnp

def sw(unroll=2, batch=True, NINF=-1e30):
  '''smith-waterman (local alignment) with gap parameter'''

  # rotate matrix for striped dynamic-programming
  def rotate(x):
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    output = {"x":jnp.full([n,m],NINF).at[i,j].set(x), "o":(jnp.arange(n)+a%2)%2}
    return output, (jnp.full(m, NINF), jnp.full(m, NINF)), (i,j)

  # compute scoring (hij) matrix
  def sco(x, lengths, gap=0, temp=1.0):

    def _soft_maximum(x, axis=None, mask=None):
      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None: return jax.nn.logsumexp(y, axis=axis)
        else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))
      return temp*_logsumexp(x/temp)

    def _cond(cond, true, false): return cond*true + (1-cond)*false
    def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))

    def _step(prev, sm):
      h2,h1 = prev   # previous two rows of scoring (hij) mtx
      h1_T = _cond(sm["o"],_pad(h1[:-1],[1,0]),_pad(h1[1:],[0,1]))

      # directions
      Align = h2 + sm["x"]
      Turn_0 = h1 + gap
      Turn_1 = h1_T + gap
      Sky = sm["x"]

      h0 = jnp.stack([Align, Turn_0, Turn_1, Sky], -1)
      h0 = _soft_maximum(h0, -1)
      return (h1,h0),h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1,:-1])
    hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
    return _soft_maximum(hij + x[1:,1:], mask=mask[1:,1:])

  # traceback (aka backprop) to get alignment
  traceback = jax.value_and_grad(sco)

  # add batch dimension
  if batch: return jax.vmap(traceback,(0,0,None,None))
  else: return traceback


def sw_affine(restrict_turns=True,
             penalize_turns=True,
             batch=True, unroll=2, NINF=-1e30):
  """smith-waterman (local alignment) with affine gap"""
  # rotate matrix for vectorized dynamic-programming


  def rotate(x):
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    output = {"x":jnp.full([n,m],NINF).at[i,j].set(x), "o":(jnp.arange(n)+a%2)%2}
    return output, (jnp.full((m,3), NINF), jnp.full((m,3), NINF)), (i,j)

  # fill the scoring matrix
  def sco(x, lengths, gap=0.0, open=0.0, temp=1.0):

    def _soft_maximum(x, axis=None, mask=None):
      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None: return jax.nn.logsumexp(y, axis=axis)
        else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))
      return temp*_logsumexp(x/temp)

    def _cond(cond, true, false): return cond*true + (1-cond)*false
    def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))

    def _step(prev, sm):
      h2,h1 = prev   # previous two rows of scoring (hij) mtxs

      Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
      Right = _cond(sm["o"], _pad(h1[:-1],([1,0],[0,0])),h1)
      Down  = _cond(sm["o"], h1,_pad(h1[1:],([0,1],[0,0])))

      # add gap penalty
      if penalize_turns:
        Right += jnp.stack([open,gap,open])
        Down += jnp.stack([open,open,gap])
      else:
        gap_pen = jnp.stack([open,gap,gap])
        Right += gap_pen
        Down += gap_pen

      if restrict_turns: Right = Right[:,:2]

      h0_Align = _soft_maximum(Align,-1)
      h0_Right = _soft_maximum(Right,-1)
      h0_Down = _soft_maximum(Down,-1)
      h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
      return (h1,h0),h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1,:-1])
    hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]

    # sink
    return _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])

  # traceback to get alignment (aka. get marginals)
  traceback = jax.value_and_grad(sco)

  # add batch dimension
  if batch: return jax.vmap(traceback,(0,0,None,None,None))
  else: return traceback

import jax 
import jax.numpy as jnp 
import haiku as hk 
#import Score_align as lddt_
import LDDT_LOSS as lddt_

key = jax.random.PRNGKey(0)

def LDDTLOSS(weights,input_data,MODEL_ETE):
    X1,mask1,res1,chain1,X2,mask2,res2,chain2,TMALN,lens,t = input_data
    col_mask = jnp.arange(X1.shape[1])[jnp.newaxis, :] >= jnp.array(lens[:,0])[:,None]
    mask__1=  ~col_mask[:, jnp.newaxis, :] * ~col_mask[:, :,jnp.newaxis]
    col_mask2 = jnp.arange(X2.shape[1])[jnp.newaxis, :] >= jnp.array(lens[:,1])[:,None]
    mask__2 =  ~col_mask2[:, jnp.newaxis, :] * ~col_mask2[:, :,jnp.newaxis]

    x1 = X1,mask1,res1,chain1
    x2 = X2,mask2,res2,chain2
    preds,sim_matrix,scores =MODEL_ETE.apply(weights,key,x1,x2,lens,t)
    scores_LDDT = lddt_.get_LDDTloss(X1[:,:,1],X2[:,:,1],preds,lens,mask__1,mask__2,t)
    aux = {"aln":preds,"sim_matrix":sim_matrix,"scores":scores,"scores_LDDT":scores_LDDT}
    loss = (1-jnp.mean(scores_LDDT))
    return loss,aux




def CrossEntropyLoss(weights, input_data,MODEL_ETE):
    X1,mask1,res1,chain1,X2,mask2,res2,chain2,TMALN,lens,t = input_data
    x1 = X1,mask1,res1,chain1
    x2 = X2,mask2,res2,chain2
    preds,sim_matrix,scores =MODEL_ETE.apply(weights,key,x1,x2,lens,t)
    aux = {"aln":preds,"sim_matrix":sim_matrix,"scores":scores}
    one_hot = jax.nn.one_hot(TMALN,TMALN.shape[1])
    loss = jnp.mean(jnp.einsum("nia,nia->n",-one_hot,jnp.log(preds+10**-4))-(jnp.einsum("nia,nia->n",1-one_hot,jnp.log(1-preds+10**-4))))
    return loss,aux

def FocalLoss(weights, input_data,MODEL_ETE):
    X1,mask1,res1,chain1,X2,mask2,res2,chain2,TMALN,lens,t = input_data
    x1 = X1,mask1,res1,chain1
    x2 = X2,mask2,res2,chain2
    preds,sim_matrix,scores =MODEL_ETE.apply(weights,key,x1,x2,lens,t)
    aux = {"aln":preds,"sim_matrix":sim_matrix,"scores":scores}
    one_hot = jax.nn.one_hot(TMALN,TMALN.shape[1])
    loss = jnp.mean(jnp.einsum("nia,nia->n",-one_hot*(1-preds)**2,jnp.log(preds+10**-4))-(jnp.einsum("nia,nia->n",(1-one_hot)*preds**2,jnp.log(1-preds+10**-4))))
    return loss,aux


def CrossEntropyLoss_CAT(weights, input_data,MODEL_ETE):
    X1,mask1,res1,chain1,X2,mask2,res2,chain2,TMALN,lens,t= input_data
    x1 = X1,mask1,res1,chain1
    x2 = X2,mask2,res2,chain2
    preds,sim_matrix,scores,seqs =MODEL_ETE.apply(weights,key,x1,x2,lens,t)
    aux = {"aln":preds,"sim_matrix":sim_matrix,"scores":scores,"seqs":seqs}
    one_hot = jax.nn.one_hot(TMALN,TMALN.shape[1])
    loss = jnp.mean(jnp.einsum("nia,nia->n",-one_hot,jnp.log(preds+10**-4))-(jnp.einsum("nia,nia->n",1-one_hot,jnp.log(1-preds+10**-4))))
    return loss,aux

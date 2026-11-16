"""MusicSparse v2: wider local bands + bounded global recurrent frequency context.
Same STFT/OLA and sample alignment as v1. No extra lookahead or growing cache.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experimental.music_sparse_v1 import MusicSparse as Base
class MusicSparseV2(Base):
 def __init__(self):
  super().__init__();self.enc=nn.ModuleList([nn.Conv1d(4,64,s,stride=s) for a,b,s in self.regions]);self.dec=nn.ModuleList([nn.ConvTranspose1d(64,4,s,stride=s) for a,b,s in self.regions])
  self.cross=nn.Conv1d(64,64,3,padding=1);self.rnn=nn.GRU(64,64,2,batch_first=True)
  self.global_in=nn.Linear(64*8,192);self.global_rnn=nn.GRU(192,192,2,batch_first=True);self.global_out=nn.Linear(192,64)
 def spectral(self,z,state=None):
  B,_,T,Q=z.shape;x=torch.cat([z.real,z.imag],1).permute(0,2,1,3).reshape(B*T,4,Q);v=x/x.square().mean((1,2),keepdim=True).add(1e-8).sqrt()
  pieces=[];sizes=[]
  for (a,b,s),enc in zip(self.regions,self.enc):
   q=enc(F.pad(v[:,:,a:b],(0,(-(b-a))%s)));pieces.append(q);sizes.append(q.shape[-1])
  q=F.silu(self.cross(torch.cat(pieces,-1)));K=q.shape[-1]
  gh=None if state is None else state[1];lh=None if state is None else state[0]
  g=F.silu(self.global_in(F.adaptive_avg_pool1d(q,8).flatten(1))).reshape(B,T,192)
  g,gh=self.global_rnn(g,gh);g=self.global_out(g).reshape(B*T,64,1)
  q=q+g;q=q.reshape(B,T,64,K).permute(0,3,1,2).reshape(B*K,T,64)
  q,lh=self.rnn(q,lh);q=q.reshape(B,K,T,64).permute(0,2,3,1).reshape(B*T,64,K)
  pred=[];offset=0
  for (a,b,s),dec,n in zip(self.regions,self.dec,sizes):pred.append(dec(q[:,:,offset:offset+n])[:,:,:b-a]);offset+=n
  mask=torch.tanh(torch.cat(pred,-1)+self.skip(v.transpose(1,2)).transpose(1,2)).reshape(B,T,4,Q).permute(0,2,1,3)
  return z*torch.complex(mask[:,:2],mask[:,2:]),(lh,gh)
 def dense_macs_per_hop(self):
  sizes=[(b-a+s-1)//s for a,b,s in self.regions];K=sum(sizes)
  enc=sum(n*64*4*s for n,(a,b,s) in zip(sizes,self.regions))
  dec=sum(n*64*4*s for n,(a,b,s) in zip(sizes,self.regions))
  return {'encoder':enc,'cross_frequency':K*64*64*3,'local_gru':K*2*3*64*(64+64),'global_projection':512*192+192*64,'global_gru':2*3*192*(192+192),'decoder':dec,'bin_skip':513*4*4}

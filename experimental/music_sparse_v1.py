"""Original SCNet-inspired cost prototype, NOT a Band-SCNet reproduction.
Three frequency regions with learned 1/4/16 sparse compression, cross-band
convolution and shared causal GRU. No temporal FFT, attention or future norm.
"""
import torch
from torch import nn
from torch.nn import functional as F
class MusicSparse(nn.Module):
 L=1024;H=256;left=768
 def __init__(self):
  super().__init__();self.regions=[(0,86,1),(86,258,4),(258,513,16)]
  self.enc=nn.ModuleList([nn.Conv1d(4,32,s,stride=s) for a,b,s in self.regions]);self.dec=nn.ModuleList([nn.ConvTranspose1d(48,4,s,stride=s) for a,b,s in self.regions])
  self.cross=nn.Conv1d(32,48,3,padding=1);self.rnn=nn.GRU(48,48,2,batch_first=True);self.skip=nn.Linear(4,4)
  self.register_buffer('win',torch.hann_window(self.L).sqrt())
 def spectral(self,z,state=None):
  # z B,stereo,T,F; all normalization local to the current frame.
  B,_,T,Q=z.shape;x=torch.cat([z.real,z.imag],1).permute(0,2,1,3).reshape(B*T,4,Q)
  scale=x.square().mean((1,2),keepdim=True).add(1e-8).sqrt();v=x/scale
  pieces=[];sizes=[]
  for (a,b,s),enc in zip(self.regions,self.enc):
   q=enc(F.pad(v[:,:,a:b],(0,(-(b-a))%s)));pieces.append(q);sizes.append(q.shape[-1])
  q=F.silu(self.cross(torch.cat(pieces,-1)));K=q.shape[-1]
  q=q.reshape(B,T,48,K).permute(0,3,1,2).reshape(B*K,T,48)
  q,state=self.rnn(q,state);q=q.reshape(B,K,T,48).permute(0,2,3,1).reshape(B*T,48,K)
  pred=[];offset=0
  for (a,b,s),dec,n in zip(self.regions,self.dec,sizes):pred.append(dec(q[:,:,offset:offset+n])[:,:,:b-a]);offset+=n
  mask=torch.cat(pred,-1)+self.skip(v.transpose(1,2)).transpose(1,2);mask=torch.tanh(mask).reshape(B,T,4,Q).permute(0,2,1,3)
  y=z*torch.complex(mask[:,:2],mask[:,2:]);return y,state
 def forward(self,x):
  B,C,N=x.shape;xp=F.pad(x,(self.left,self.L));fr=xp.unfold(-1,self.L,self.H);z=torch.fft.rfft(fr*self.win);y,_=self.spectral(z)
  f=torch.fft.irfft(y,n=self.L)*self.win;T=f.shape[-2];length=(T-1)*self.H+self.L
  raw=F.fold(f.reshape(B*C,T,self.L).transpose(1,2),(1,length),(1,self.L),stride=(1,self.H)).reshape(B,C,length)
  den=F.fold(self.win.square()[None,:,None].expand(1,-1,T),(1,length),(1,self.L),stride=(1,self.H)).reshape(1,1,length)
  return (raw/den.clamp_min(1e-8))[:,:,self.left:self.left+N]
 def init_state(self,B=1):
  z=self.win.new_zeros;return [z(B,2,self.left),None,z(B,2,self.left),z(1,1,self.left)]
 def step(self,x,state):
  history,h,tail,den=state;fr=torch.cat([history,x],-1);z=torch.fft.rfft(fr*self.win).unsqueeze(2);y,h=self.spectral(z,h)
  raw=torch.fft.irfft(y[:,:,0],n=self.L)*self.win;env=self.win.square()[None,None,:].expand(1,1,-1)
  raw=raw+F.pad(tail,(0,self.H));env=env+F.pad(den,(0,self.H));out=raw[:,:,:self.H]/env[:,:,:self.H].clamp_min(1e-8)
  return out,[fr[:,:,-self.left:],h,raw[:,:,self.H:],env[:,:,self.H:]]

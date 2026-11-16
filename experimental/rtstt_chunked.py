"""Causal multi-hop wrapper sharing RT-STT frame/state logic, for efficient GPU validation.
Deployment comparison remains one hop512 per ONNX call. No state reset within a track.
"""
import torch
import torch.nn.functional as F

def step_chunk(m,block,state=None):
 B,ch,T=block.shape
 assert T%m.hop==0 and T>=m.hop and not m.training
 if state is None:state={'inbuf':block.new_zeros(B,ch,m.left),'net':None,'carry':block.new_zeros(B,m.ns,ch,m.left)}
 buf=torch.cat([state['inbuf'],block],-1);frames=buf.unfold(-1,m.n_fft,m.hop)
 X=m._stft_frames(frames);logits,netstate=m.net(X,state['net']);fr=m._istft_frames(m._apply_mask(logits,X));nf=fr.shape[3]
 raw=F.fold(fr.reshape(B*m.ns*ch,nf,m.n_fft).transpose(1,2),output_size=(1,T+m.left),kernel_size=(1,m.n_fft),stride=(1,m.hop)).reshape(B,m.ns,ch,-1)
 raw=raw+F.pad(state['carry'],(0,T))
 return raw[...,:T],{'inbuf':buf[...,-m.left:],'net':netstate,'carry':raw[...,T:]}

@torch.no_grad()
def infer_track(m,mix,device,chunk=32768):
 """mix numpy (T,2), return CPU numpy (4,T,2), exact fixed-delay removal."""
 import numpy as np,math
 T=len(mix);out=np.empty((m.ns,T,m.ach),np.float32);state=None
 for start in range(0,math.ceil((T+m.left)/chunk)*chunk,chunk):
  data=np.zeros((m.ach,chunk),np.float32);end=min(T,start+chunk)
  if end>start:data[:,:end-start]=mix[start:end].T
  y,state=step_chunk(m,torch.from_numpy(data)[None].to(device),state);y=y[0].cpu().numpy()
  lo=max(0,start-m.left);hi=min(T,start+chunk-m.left)
  if hi>lo:out[:,lo:hi,:]=y[:,:,lo-(start-m.left):hi-(start-m.left)].transpose(0,2,1)
 return out

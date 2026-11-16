"""RT-STT paper-graph adaptation v2; not an exact official reproduction.
Corrects legacy rtstt.py's undocumented frequency downsampling and output mask:
Fig.1 keeps F throughout and places source softmax before the decoder.
Decoder output is interpreted as complex spectra (DTTNet convention); this
interpretation, BN, bottleneck8, hidden32, and sqrt-Hann remain documented choices.
The first adaptation (rtstt.py) is kept as the base class.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .rtstt import RTSTT as LegacyRTSTT

class RTSTT(LegacyRTSTT):
 def __init__(self,architecture_revision='paper_graph_v2',**kwargs):
  assert architecture_revision=='paper_graph_v2'
  kwargs.setdefault('freq_ds',1)
  assert kwargs['freq_ds']==1,'Paper graph keeps F; no frequency downsampling'
  super().__init__(**kwargs)
  assert self.n_fft==1024 and self.hop==512 and self.Fb==384 and self.ns==4
  self.us=nn.Conv2d(self.ns*2*self.g,self.ns*self.g,1)
  self.cfg['architecture_revision']=architecture_revision
 def net(self,X,state=None):
  st=state if state is not None else {}
  x=F.relu(self.first_n(self.first(X)));e,s_enc=self.enc(x,st.get('enc'))
  x=F.relu(self.ds_n(self.ds(e)));x,s_lat=self.lat(x,st.get('lat'));s_sp=[]
  for i,sp in enumerate(self.sp):
   x,s=sp(x,st.get('sp',[None]*len(self.sp))[i]);s_sp.append(s)
  x=self.expand(x);B,_,T,Fb=x.shape
  x=torch.softmax(x.reshape(B,self.ns,2*self.g,T,Fb),dim=1).reshape(B,self.ns*2*self.g,T,Fb)
  x=F.relu(self.us_n(self.us(x)));x=x*e.repeat(1,self.ns,1,1)
  x,s_dec=self.dec(x,st.get('dec'))
  return self.final(x),{'enc':s_enc,'lat':s_lat,'sp':s_sp,'dec':s_dec}
 def _apply_mask(self,logits,X):
  # Name retained only for compatibility with shared STFT/streaming wrapper.
  B,_,T,Fb=logits.shape
  return logits.reshape(B,self.ns,self.C,T,Fb)

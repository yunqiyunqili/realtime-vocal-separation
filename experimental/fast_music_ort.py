"""Real-valued DFT/OLA MusicSparse export; fixed 512 input, recurrent state retained."""
import io,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
class Step(nn.Module):
    def __init__(self,m):
        super().__init__();self.m=m;self.K=sum((b-a+s-1)//s for a,b,s in m.regions)
        n=torch.arange(m.L,dtype=torch.float64);k=torch.arange(m.L//2+1,dtype=torch.float64)
        a=2*math.pi*n[:,None]*k[None,:]/m.L
        self.register_buffer('co',a.cos().float());self.register_buffer('si',a.sin().float())
        scale=torch.full((m.L//2+1,),2/m.L);scale[0]=scale[-1]=1/m.L
        self.register_buffer('ico',(self.co*scale).T);self.register_buffer('isi',(self.si*scale).T)
    def initial(self):
        m=self.m
        return [torch.zeros(1,2,m.left),torch.zeros(2,self.K,64),torch.zeros(2,1,192),torch.zeros(1,2,m.left),torch.zeros(1,1,m.left)]
    def forward(self,block,hist,lh,gh,tail,den):
        m=self.m;ys=[]
        for j in range(512//m.H):
            fr=torch.cat([hist,block[:,:,j*m.H:(j+1)*m.H]],-1);f=fr*m.win
            re=f@self.co;im=-(f@self.si);v=torch.cat([re,im],1);v=v/(v.square().mean((1,2),keepdim=True)+1e-8).sqrt()
            pieces=[];sizes=[]
            for (a,b,s),enc in zip(m.regions,m.enc):
                q=enc(F.pad(v[:,:,a:b],(0,(-(b-a))%s)));pieces.append(q);sizes.append(q.shape[-1])
            q=F.silu(m.cross(torch.cat(pieces,-1)))
            pooled=torch.cat([q[:,:,i*self.K//8:math.ceil((i+1)*self.K/8)].mean(-1,keepdim=True) for i in range(8)],-1)
            g=F.silu(m.global_in(pooled.flatten(1))).reshape(1,1,192);g,gh=m.global_rnn(g,gh)
            q=q+m.global_out(g).reshape(1,64,1);q=q.permute(2,0,1);q,lh=m.rnn(q,lh);q=q.permute(1,2,0)
            pred=[];off=0
            for (a,b,s),dec,n in zip(m.regions,m.dec,sizes):pred.append(dec(q[:,:,off:off+n])[:,:,:b-a]);off+=n
            mask=torch.tanh(torch.cat(pred,-1)+m.skip(v.transpose(1,2)).transpose(1,2))
            yr=re*mask[:,:2]-im*mask[:,2:];yi=re*mask[:,2:]+im*mask[:,:2]
            raw=(yr@self.ico-yi@self.isi)*m.win+F.pad(tail,(0,m.H))
            env=m.win.square()[None,None,:]+F.pad(den,(0,m.H))
            ys.append(raw[:,:,:m.H]/env[:,:,:m.H].clamp_min(1e-8))
            hist=fr[:,:,-m.left:];tail=raw[:,:,m.H:];den=env[:,:,m.H:]
        return torch.cat(ys,-1),hist,lh,gh,tail,den
def make(m,path):
    import onnxruntime as ort
    core=Step(m).eval();initial=core.initial();names=['block','hist','lh','gh','tail','den']
    torch.onnx.export(core,(torch.zeros(1,2,512),*initial),str(path),input_names=names,opset_version=17,dynamo=False)
    so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
    so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    sess=ort.InferenceSession(str(path),so,providers=['CPUExecutionProvider'])
    def step(b,st):
        y=sess.run(None,dict(zip(names,[b[None].numpy(),*st])));return torch.from_numpy(y[0][0]),y[1:]
    return step,lambda:[t.numpy().copy() for t in initial]

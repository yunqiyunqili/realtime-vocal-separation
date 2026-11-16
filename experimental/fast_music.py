"""MusicSparse v2 variants. 'fft512' = model C in the README (FFT 512, hop 128, 11.6 ms algorithmic latency);
'stride8' = a coarser high-band compression variant (23.2 ms)."""
import torch
from torch import nn
from experimental.music_sparse_v2 import MusicSparseV2
class FastMusic(MusicSparseV2):
    def __init__(self,variant='original'):
        super().__init__();self.variant=variant
        if variant=='stride8':self.regions=[(0,86,1),(86,258,4),(258,513,8)]
        elif variant=='fft512':
            self.L=512;self.H=128;self.left=384
            self.regions=[(0,43,1),(43,129,4),(129,257,16)]
            self.win=torch.hann_window(self.L).sqrt()
        elif variant!='original':raise ValueError(variant)
        if variant!='original':
            self.enc=nn.ModuleList([nn.Conv1d(4,64,s,stride=s) for a,b,s in self.regions])
            self.dec=nn.ModuleList([nn.ConvTranspose1d(64,4,s,stride=s) for a,b,s in self.regions])
    def dense_macs_per_hop(self):
        d=super().dense_macs_per_hop();d['bin_skip']=(self.L//2+1)*16;return d

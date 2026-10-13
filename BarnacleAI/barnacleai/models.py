"""A lightweight pixelwise CNN to learn barnacle-center heatmaps."""
import torch
from torch import nn

class BarnacleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(inplace=True),
                                 nn.Conv2d(b,b,3,padding=1),nn.ReLU(inplace=True))
        self.enc1=block(3,8)
        self.pool=nn.MaxPool2d(2)
        self.enc2=block(8,16)
        self.bottleneck=block(16,24)
        self.up2=block(24+16,16)
        self.up1=block(16+8,8)
        self.out=nn.Conv2d(8,1,1)

    def forward(self,x):
        import torch.nn.functional as F
        s1=self.enc1(x)
        s2=self.enc2(self.pool(s1))
        z=self.bottleneck(self.pool(s2))
        z=F.interpolate(z,size=s2.shape[-2:],mode='bilinear',align_corners=False)
        z=self.up2(torch.cat([z,s2],dim=1))
        z=F.interpolate(z,size=s1.shape[-2:],mode='bilinear',align_corners=False)
        return self.out(self.up1(torch.cat([z,s1],dim=1)))

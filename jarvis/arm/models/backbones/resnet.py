import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
import torchvision
from torchvision import transforms as T
from typing import Dict, Optional, Union, List, Any, Tuple

class CustomResNet(nn.Module):
    
    def __init__(self, version: str = '18', out_dim: int = 1024, pooling: bool = False, **kwargs):
        super().__init__()
        if version == '18':
            self.model = torchvision.models.resnet18(pretrained=True)
        elif version == '50':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif version == '101':
            self.model = torchvision.models.resnet101(pretrained=True)
        
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.final_layer = nn.Sequential(
            nn.GELU(), 
            nn.Conv2d(512, out_dim, 1)
        )
        if pooling:
            self.pooling_layer = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, imgs, **kwargs):
        imgs = imgs / 255.
        B, T = imgs.shape[:2]
        if imgs.shape[-1] == 3:
            x = rearrange(imgs, 'b t h w c -> (b t) c h w')
        else:
            x = rearrange(imgs, 'b t c h w -> (b t) c h w')
        x = self.model(x)
        if hasattr(self, 'pooling_layer'):
            x = self.pooling_layer(x)
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c 1 1 -> b t c', b=B, t=T)
        else:
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x


if __name__ == '__main__':
    model = CustomResNet(
        version='18', 
        out_dim=1024, 
        pooling=False,
    ).to("cuda")
    
    B, T = 4, 128
    example = torch.rand(B, T, 3, 224, 224).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
    
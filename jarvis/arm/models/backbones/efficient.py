import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
from typing import Dict, Optional, Union, List, Any, Tuple
from jarvis.arm.utils.efficientnet_lib import EfficientNet

class CustomEfficientNet(nn.Module):
    
    def __init__(
        self, 
        version: str, 
        resolution: int = 224, 
        out_dim: int = 1024, 
        pooling: bool = False, 
        **kwargs, 
    ) -> None:
        super().__init__()
        self.version = version
        self.resoulution = resolution
        self.out_dim = out_dim
        
        self.model = EfficientNet.from_pretrained(version)
        
        if 'b0' in version:
            self.mid_dim = 1280
        elif 'b4' in version:
            self.mid_dim = 1792
        
        if resolution == 360:
            self.feat_reso = (11, 11)
        elif resolution == 224:
            self.feat_reso = (7, 7)
        elif resolution == 128:
            self.feat_reso = (4, 4)

        self.final_layer = nn.Sequential(
            nn.GELU(), 
            nn.Conv2d(self.mid_dim, out_dim, 1)
        )
        
        if pooling:
            self.pooling_layer = nn.AdaptiveAvgPool2d(1)

    def forward(self, imgs, **kwargs): 
        '''
        :params imgs: shape of (B, T, 3, H, W)
        :returns: shape of (B, T, C, R, R)
        '''
        imgs = imgs / 255.
        B, T = imgs.shape[:2]
        if imgs.shape[-1] == 3:
            x = rearrange(imgs, 'b t h w c -> (b t) c h w')
        else:
            x = rearrange(imgs, 'b t c h w -> (b t) c h w')
        x = self.model.extract_features(x)
        if hasattr(self, 'pooling_layer'):
            x = self.pooling_layer(x)
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c 1 1 -> b t c', b=B, t=T)
        else:
            x = self.final_layer(x)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return x

if __name__ == '__main__':
    model = CustomEfficientNet(
        version='efficientnet-b0', 
        resolution=128, 
        out_dim=1024, 
        pooling=False,
    ).to("cuda")
    B, T = 4, 128
    example = torch.rand(B, T, 3, 128, 128).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
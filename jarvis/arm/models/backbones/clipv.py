import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from einops import rearrange
from typing import Dict, Optional, Union, List, Any, Tuple
from transformers import CLIPVisionModel
from transformers import logging
logging.set_verbosity_error()

class CustomCLIPv(nn.Module):
    
    def __init__(self, version: str = "openai/clip-vit-base-patch32", out_dim: int = 1024, freeze: bool = False, **kwargs):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(version)
        self.final_layer = nn.Linear(self.vision_encoder.config.hidden_size, out_dim)
        self.transform = T.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, imgs, **kwargs):
        B, T = imgs.shape[:2]
        imgs = imgs / 255.
        if imgs.shape[-1] == 3:
            x = rearrange(imgs, 'b t h w c -> (b t) c h w')
        else:
            x = rearrange(imgs, 'b t c h w -> (b t) c h w')
        x = self.transform(x)
        x = self.vision_encoder(x).last_hidden_state
        x = self.final_layer(x) # x: (B*T, num_tokens, out_dim)
        x = x[:, 1:, :] # remove the CLS token
        r = int (np.sqrt(x.shape[1]))
        assert r * r == x.shape[1]
        x = rearrange(x, '(b t) (h w) c -> b t c h w', b=B, t=T, h=r, w=r)
        return x

if __name__ == '__main__':
    model = CustomCLIPv().to("cuda")
    B, T = 4, 128
    example = torch.rand(B, T, 224, 224, 3).to("cuda")
    
    output = model(example)
    
    print(f"{output.shape = }")
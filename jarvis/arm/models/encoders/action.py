import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Union, List, Any
from collections import OrderedDict
from gymnasium import spaces
from jarvis.arm.models.utils import FeedForward



    
            
        
    
    
        

class Action(nn.Module):
    
    def __init__(self, hidsize: int, action_space: Dict, n_layer=2, **kwargs):
        super().__init__()
        self.act_emb = nn.ModuleDict({
            name: nn.Embedding(act.n, hidsize) for name, act in action_space.items()
        })
        self.act_names = sorted(list(action_space.keys()))
        self.updim = nn.Sequential(
            nn.GELU(), nn.Linear(hidsize*len(self.act_names), hidsize)
        )
        self.feedforwards = nn.ModuleList([
            FeedForward(hidsize, mult=2) for i in range(n_layer)
        ])
    
    def forward(self, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute joint action embeddings. 
        :params actions: 
        """
        acts = []
        for act_name in self.act_names:
            acts.append(self.act_emb[act_name](actions[act_name]))
        x = torch.cat(acts, dim=-1)
        x = self.updim(x)
        for ffn in self.feedforwards:
            x = ffn(x) + x
        return x

if __name__ == '__main__':
    pass 
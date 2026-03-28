import random
import numpy as np
import torch
from torch import nn
from rich.console import Console
from typing import Dict, Optional, Union, List, Any, Tuple
from jarvis.arm.utils.transformers import GPT 
from vector_quantize_pytorch import VectorQuantize, FSQ
from einops import rearrange

import torch.distributions as distr
from jarvis.arm.models.normal import StableNormal
from jarvis.arm.models.mixture_same_family import ReparametrizedMixtureSameFamily

class MixtureSpace(nn.Module):
    
    def __init__(self, input_dim: int, latent_hidsize: int, n_components: int = 1, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.temperature = temperature
        self.enc_mu = nn.Linear(input_dim, latent_hidsize * n_components)
        self.enc_logvar = nn.Linear(input_dim, latent_hidsize * n_components)
        self.enc_pi = nn.Linear(input_dim, n_components)
        print(f"{self.temperature = }")
    
    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == 1, 'MixtureSpace expects input of shape (B, 1, D)'
        x = x[:, 0, :]
        mu = self.enc_mu(x) # B, n_components * D
        log_var = self.enc_logvar(x) # B, n_components * D
        pi = self.enc_pi(x) / self.temperature # B, n_components
        locs = rearrange(mu, 'b (n d) -> b n d', n=self.n_components)
        stds = rearrange(torch.exp(log_var * 0.5), 'b (n d) -> b n d', n=self.n_components)
        mixture_probs = torch.softmax(pi, dim=-1) # B, n_components
        
        mixture = distr.Categorical(probs=mixture_probs) 
        components = distr.Independent(StableNormal(loc=locs, scale=stds), 1)
        mog = ReparametrizedMixtureSameFamily(mixture_distribution=mixture, component_distribution=components)
        z = mog.rsample()
        
        return {
            'z': z[:, None, ...], 
            'mu': mu[:, None, ...], 
            'log_var': log_var[:, None, ...], 
            'pi': pi[:, None, ...], 
        }

class VAE_Space(nn.Module):
    
    def __init__(self, input_dim: int, latent_hidsize: int, **kwargs):
        super().__init__()
        self.encode_mu  = nn.Linear(input_dim, latent_hidsize)
        self.encode_var = nn.Linear(input_dim, latent_hidsize)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.encode_mu(x)
        log_var = self.encode_var(x)
        if self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return {
            'z': z, 'mu': mu, 'log_var': log_var, 
        }

class VQ_Space(nn.Module):
    
    def __init__(self, input_dim: int, num_slots: int, **kwargs):
        super().__init__()
        quantizers = []
        for i in range(num_slots):
            quantizers.append(VectorQuantize(dim=input_dim, **kwargs))
        self.quantizers = nn.ModuleList(quantizers)
    
    def forward(self, x):
        '''
        x: shape of (B, num_slots, input_dim)
        '''
        quantized = []
        commitment_loss = []
        for i, quantizer in enumerate(self.quantizers):
            y = x[:, i, :]
            zi, indices, lossi = quantizer(y)
            quantized.append(zi)
            commitment_loss.append(lossi)
        quantized = torch.stack(quantized, dim=1)
        commitment_loss = torch.stack(commitment_loss, dim=0).mean()
        return {
            'z': quantized, 'commitment_loss': commitment_loss, 
        }


class FSQ_Space(nn.Module):
    
    def __init__(self, input_dim: int, levels: List[int], show_usage: bool = False, **kwargs):
        super().__init__()
        self.quantizer = FSQ(levels, dim=input_dim)
        self.show_usage = show_usage
        self.weight = 1.0
        if self.show_usage:
            self.usage_list = []

    def forward(self, x):
        quantized, indices = self.quantizer(x)
        if self.weight < 1.0:
            quantized = self.weight * quantized + (1 - self.weight) * x
        
        if self.show_usage:
            self.usage_list.append(indices)

        return {
            'z': quantized, 'indices': indices, 
        }
    
    def get_usage_info(self):
        res = torch.cat(self.usage_list, dim=0).flatten().cpu().numpy()
        self.usage_list = []
        return res

def build_latent_space(input_dim: int, **latent_space_kwargs):
    space_type = latent_space_kwargs.pop('type')
    if space_type == 'VAE':
        return VAE_Space(input_dim, **latent_space_kwargs)
    elif space_type == 'Mixture':
        return MixtureSpace(input_dim, **latent_space_kwargs)
    elif space_type == 'VQ-VAE':
        return VQ_Space(input_dim, 1, **latent_space_kwargs)
    elif space_type == 'FSQ':
        return FSQ_Space(input_dim, **latent_space_kwargs)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    pass
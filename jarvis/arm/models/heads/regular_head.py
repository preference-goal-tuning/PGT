import random
from typing import Dict, Optional, Union, List, Any
from rich.console import Console

import numpy as np
import torch
from torch import nn
from torchmetrics.regression import ExplainedVariance
from einops import rearrange, repeat

def route_to(obj: Dict, path: Union[str, List[str]]):
    if isinstance(path, str):
        path = path.split('.')
    for key in path:
        obj = obj[key]
    return obj

class KLRegularHead(nn.Module):
    
    def __init__(
        self, 
        alias: str = "",
        weight: float = 1.0, 
        KL_threshold: float = 0., 
        enable_KL_thresholding: bool = False,
        alpha: float = 0.8, 
        enable_KL_balancing: bool = False,
        q_dist_path: str = '', 
        p_dist_path: str = '', 
        **kwargs
    ) -> None:
        super().__init__()
        self.alias = alias
        self.weight = weight
        self.KL_threshold = KL_threshold
        self.enable_KL_thresholding = enable_KL_thresholding
        if self.enable_KL_thresholding:
            Console().log(f"Enable KL thresholding, threshold = {KL_threshold}.")
        self.alpha = alpha
        self.enable_KL_balancing = enable_KL_balancing
        if self.enable_KL_balancing:
            Console().log(f"Enable KL balancing, alpha = {self.alpha}.")
        self.q_dist_path = q_dist_path
        self.p_dist_path = p_dist_path
        self.explained_variance = ExplainedVariance()
        
    def forward(self, latents, **kwargs) -> Dict[str, torch.Tensor]:
        return {
            'q': route_to(latents, self.q_dist_path), 
            'p': route_to(latents, self.p_dist_path),
        }
    
    def kl_divergence(self, q_mu, q_log_var, p_mu, p_log_var):
        DKL = -0.5 * torch.sum(
            1 + (q_log_var - p_log_var) - (q_log_var - p_log_var).exp() - (q_mu - p_mu).pow(2) / p_log_var.exp(), dim=(-1, -2)
        ) # shape: (B)
        oDKL = DKL
        if self.enable_KL_thresholding:
            DKL = torch.max(DKL, torch.ones_like(DKL) * self.KL_threshold)
        return DKL, oDKL
    
    def filter_KL_by_condition(self, condition_name: List[str], KL_loss: torch.Tensor, prefix: str = '') -> Dict[str, torch.Tensor]:
        res = {}
        for idx, name in enumerate(condition_name):
            if name not in res:
                res[name] = []
            res[name] += [KL_loss[idx]]
        res = {f"{prefix}_{k}_KL_ori": torch.stack(v, dim=0) for k, v in res.items()}
        return res
    
    def loss(self, obs, pred, mask=None, **kwargs):
        q_mu, q_log_var = pred['q']['mu'], pred['q']['log_var']
        p_mu, p_log_var = pred['p']['mu'], pred['p']['log_var']
        assert len(q_mu.shape) == 3 and len(p_mu.shape) == 3, "latent's shape should be [B, T, D]"
        if self.enable_KL_balancing:
            pDKL, oDKL = self.kl_divergence(q_mu.detach(), q_log_var.detach(), p_mu, p_log_var)
            qDKL, _    = self.kl_divergence(q_mu, q_log_var, p_mu.detach(), p_log_var.detach())
            DKL  = self.alpha * pDKL + (1-self.alpha) * qDKL
        else:
            DKL, oDKL = self.kl_divergence(q_mu, q_log_var, p_mu, p_log_var)
        
        res = {
            f'{self.alias}_KL': DKL, 
            f'{self.alias}_KL_original': oDKL,
            f'{self.alias}_KL_weight': self.weight, 
            f'{self.alias}_KL_loss': DKL * self.weight, 
        }
        
        if (condition_info := obs.get('condition_info', None)) is not None:
            res.update(
                self.filter_KL_by_condition(
                    condition_name=condition_info['condition_name'], KL_loss=oDKL
                )
            )
        
        return res
    
        
    #     ##? gradient for q distribution (q can be text or video)
        
        
        
        
        

import torch.distributions as distr
from jarvis.arm.models.normal import StableNormal
from jarvis.arm.models.mixture_same_family import ReparametrizedMixtureSameFamily

class MixtureKLHead(KLRegularHead):
    
    def __init__(self, *args, n_sample: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample = n_sample
    
    def build_mog(self, pi, mu, log_var):
        n_components = pi.shape[-1]
        pi, mu, log_var = pi[:, 0], mu[:, 0], log_var[:, 0]
        log_var = torch.clamp(log_var, -5.0)
        locs = rearrange(mu, 'b (n d) -> b n d', n=n_components)
        stds = rearrange(torch.exp(0.5 * log_var), 'b (n d) -> b n d', n=n_components)
        mixture_probs = torch.softmax(pi, dim=-1)
        self.record_mixture_probs = mixture_probs # for wandb logger
        mixture = distr.Categorical(probs=mixture_probs) 
        components = distr.Independent(StableNormal(loc=locs, scale=stds), 1)
        mog = ReparametrizedMixtureSameFamily(mixture_distribution=mixture, component_distribution=components)
        return mog
    
    def kl_divergence(self, q_mog, p_mog):
        points = q_mog.rsample(sample_shape=(self.n_sample,)) # shape: (n_sample, B, D)
        log_q = q_mog.log_prob(points)
        log_p = p_mog.log_prob(points)
        DKL = (log_q - log_p).mean(dim=0)
        return DKL
    
    def loss(self, obs, pred, mask=None, **kwargs):
        # get mixture distributions
        p_mog = self.build_mog(pred['p']['pi'], pred['p']['mu'], pred['p']['log_var']) 
        q_mog = self.build_mog(pred['q']['pi'], pred['q']['mu'], pred['q']['log_var']) 
        assert self.enable_KL_balancing == False, "MixtureKLHead does not support KL balancing."
        
        DKL = self.kl_divergence(q_mog, p_mog)
        
        res = {
            f'{self.alias}_KL': DKL, 
            f'{self.alias}_KL_weight': self.weight, 
            f'{self.alias}_KL_loss': DKL * self.weight, 
        }
        
        res['mixture_probs_max'] = self.record_mixture_probs.max(dim=-1).values
        res['mixture_probs_min'] = self.record_mixture_probs.min(dim=-1).values
        
        if (condition_info := obs.get('condition_info', None)) is not None:
            res.update(
                self.filter_KL_by_condition(
                    condition_name=condition_info['condition_name'], KL_loss=DKL
                )
            )
        
        return res
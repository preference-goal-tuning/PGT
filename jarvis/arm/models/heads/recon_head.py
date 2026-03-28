import random
from typing import Dict, Literal, Optional, Union, List, Any, Literal
from rich.console import Console

import numpy as np
import torch
from torch import nn

class MinecraftReconHead(nn.Module):
    
    def __init__(
        self, 
        alias: str = "minecraft", 
        weight: float = 1.0, 
        reduction: Literal['sum', 'mean'] = 'sum', 
        **kwargs
    ) -> None:
        super().__init__()
        self.alias = alias
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, latents, useful_heads, **kwargs):
        return {
            'pi_head': useful_heads['pi_head'],
            'pi_latent': latents['pi_latent'],
        }
    
    def loss(self, obs, pred, mask=None, **kwargs):
        pi_head = pred['pi_head']
        pi_logits = pi_head(pred['pi_latent'])
        nll_BC, nll_buttons, nll_camera, entropy = self.compute_hierarchical_logp(
            action_head=pi_head,
            agent_action=obs['minecraft_action'],
            pi_logits=pi_logits, 
            mask=mask, 
            reduction=self.reduction
        )
        
        sample_mask = torch.tensor([env == self.alias for env in obs['env']], device=nll_BC.device) * 1.0
        nll_BC = nll_BC * sample_mask
        nll_buttons = nll_buttons * sample_mask
        nll_camera = nll_camera * sample_mask
        entropy = entropy * sample_mask
        
        res = {
            f'({self.alias}) nll_BC': nll_BC,
            f'({self.alias}) nll_buttons': nll_buttons,
            f'({self.alias}) nll_camera': nll_camera,
            f'({self.alias}) recon_weight': self.weight,
            f'({self.alias}) recon_loss': self.weight * nll_BC,
            f'({self.alias}) entropy': entropy, 
        }
        return res

    def compute_hierarchical_logp(
        self, 
        action_head: nn.Module, 
        agent_action: Dict, 
        pi_logits: Dict, 
        mask: Optional[torch.Tensor] = None,
        reduction: Literal['mean', 'sum'] = 'sum',
        eps: float = 1e-6, 
    ):
        log_prob = action_head.logprob(agent_action, pi_logits, return_dict=True)
        entropy  = action_head.entropy(pi_logits, return_dict=True)
        camera_mask = (agent_action['camera'] != 60).float().squeeze(-1)
        if mask is None:
            mask = torch.ones_like(camera_mask)
        logp_buttons = (log_prob['buttons'] * mask).sum(-1)
        logp_camera  = (log_prob['camera'] * mask * camera_mask).sum(-1)
        entropy_buttons = (entropy['buttons'] * mask).sum(-1)
        entropy_camera  = (entropy['camera'] * mask * camera_mask).sum(-1)
        if reduction == 'mean':
            logp_buttons = logp_buttons / (mask.sum(-1) + eps)
            logp_camera  = logp_camera / ((mask * camera_mask).sum(-1) + eps)
            entropy_buttons = entropy_buttons / (mask.sum(-1) + eps)
            entropy_camera  = entropy_camera / ((mask * camera_mask).sum(-1) + eps)
        logp_bc = logp_buttons + logp_camera
        entropy = entropy_buttons + entropy_camera
        return -logp_bc, -logp_buttons, -logp_camera, entropy


class NLLReconHead(nn.Module):
    
    def __init__(
        self, 
        alias: str, 
        weight: float = 1.0, 
        reduction: Literal['sum', 'mean'] = 'sum', 
        **kwargs
    ) -> None:
        super().__init__()
        self.alias = alias
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, latents, useful_heads, **kwargs):
        return {
            'pi_head': useful_heads['auxiliary_pi_heads'][self.alias],
            'pi_latent': latents['pi_latent'],
        }
    
    def loss(self, obs, pred, mask=None, **kwargs):
        
        action = obs[f'{self.alias}_action']
        pi_head = pred['pi_head']
        pi_logits = pi_head(pred['pi_latent'])
        nll_BC = -pi_head.logprob(action, pi_logits)
        entropy = pi_head.entropy(pi_logits)
        B, T = nll_BC.shape

        if mask is None:
            mask = torch.ones((B, T), device=nll_BC.device)
        nll_BC = (nll_BC * mask).sum(-1)
        entropy = (entropy * mask).sum(-1)
        
        if self.reduction == 'mean':
            nll_BC = nll_BC / (mask.sum(-1) + eps)
            entropy = entropy / (mask.sum(-1) + eps)
        
        sample_mask = torch.tensor([env == self.alias for env in obs['env']], device=nll_BC.device) * 1.0
        nll_BC = nll_BC * sample_mask
        entropy = entropy * sample_mask
        
        res = {
            f'({self.alias}) nll_BC': nll_BC,
            f'({self.alias}) recon_weight': self.weight,
            f'({self.alias}) recon_loss': self.weight * nll_BC,
            f'({self.alias}) entropy': entropy, 
        }
        return res
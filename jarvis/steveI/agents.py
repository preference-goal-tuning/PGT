import functools
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import rich
import torch
from jarvis.arm.models.agents.base_agent import BaseAgent, recursive_tensor_op
from omegaconf import DictConfig
from jarvis.steveI.steveI_lib.utils.mineclip_agent_env_utils import make_agent
from jarvis.steveI.steveI_lib.utils.embed_utils import get_prior_embed
from jarvis.steveI.steveI_lib.config import MINECLIP_CONFIG, PRIOR_INFO
import jarvis.steveI.steveI_lib.mineclip_code.load_mineclip as load_mineclip
from jarvis.steveI.steveI_lib.data.text_alignment.vae import load_vae_model
from jarvis.assembly.marks import MarkBase
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.steveI.steveI_lib.embed_conditioned_policy import tree_map

class SteveITextAgent(BaseAgent):
    def __init__(self, policy_config: DictConfig, **kwargs):
        super().__init__()
        self.policy_cfg = policy_config

        self.cond_scale = self.policy_cfg.text_cond_scale
        mineclip_config = MINECLIP_CONFIG
        mineclip_config['ckpt']['path'] = self.policy_cfg.mineclip_weights

        prior_info = PRIOR_INFO
        prior_info['model_path'] = self.policy_cfg.prior_weights
        self.prior = load_vae_model(prior_info, device=torch.device("cpu"))
        self.policy = make_agent(self.policy_cfg.in_model, self.policy_cfg.in_weights, cond_scale=self.cond_scale, device=torch.device("cpu")).policy

        self.mineclip = load_mineclip.load(mineclip_config, device=torch.device("cpu"))

        self.prompt_embeds = {}

    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        return recursive_tensor_op(lambda x : x.to(self.device), self.policy.initial_state(2))

    @torch.no_grad()
    def get_embed(self, prompt: str):
        if prompt not in self.prompt_embeds:
            self.prompt_embeds[prompt] = get_prior_embed(prompt, self.mineclip, self.prior, self.device)
        return torch.from_numpy(self.prompt_embeds[prompt]).to(self.device)

    # @property
    def action_head(self):
        return self.policy.pi_head
    
    @functools.lru_cache(maxsize=None)
    def direct_read_latent(self, given_latent_file) -> torch.Tensor:
        with open(given_latent_file, 'rb') as f:
            latent = pickle.load(f)
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).to(self.device).view((1, 512))
        elif isinstance(latent, torch.Tensor):
            latent = latent.to(self.device).view((1, 512))
        else:
            raise TypeError('Given latent should be np.ndarray or torch.Tensor, got {}'.format(type(latent)))
        return latent

    @torch.cuda.amp.autocast()
    def forward(self, obs: Dict[str, torch.Tensor], state_in: Optional[List[torch.Tensor]], first: Optional[torch.Tensor], **kwargs):
        assert first is None or not first.any()
        B, T = obs['img'].shape[:2]
        assert B == 1 and T == 1
        first = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        given_latent_file = obs['obs_conf'].get('given_latent', None)
        while isinstance(given_latent_file, list):
            given_latent_file = given_latent_file[0]
        if isinstance(given_latent_file, str) and given_latent_file.endswith('.pkl'):
            latent = self.direct_read_latent(given_latent_file=given_latent_file)
            rich.print('[blue]>>>> Use given latent!')
        else:
            control_text = obs['text'][0][0]
            latent = self.get_embed(control_text)
            rich.print(f'[red]<<<< No given! Use {control_text}')
        obs['mineclip_embed'] = latent

        obs = {"img": obs["img"], "mineclip_embed": obs["mineclip_embed"]}
        obs = tree_map(lambda x: torch.cat([x, x], dim=0), obs)
        obs["mineclip_embed"][1] = torch.zeros_like(obs["mineclip_embed"][1])
        first = torch.cat([first, first], dim=0)


        state_in = recursive_tensor_op(lambda x: x.squeeze(0), state_in)

        (pd, vpred, _), state_out = self.policy(obs=obs, first=first, state_in=state_in)
        if self.cond_scale is not None:
            pd = tree_map(lambda x: (((1 + self.cond_scale) * x[0]) - (self.cond_scale * x[1])).unsqueeze(0), pd)

        state_out = recursive_tensor_op(lambda x: x.unsqueeze(0), state_out)

        return {"pi_logits": pd, "vpred": vpred}, state_out, {}



class SteveIVisualAgent(BaseAgent):
    def __init__(self, policy_config: DictConfig, **kwargs):
        super().__init__()
        self.policy_cfg = policy_config

        self.cond_scale = self.policy_cfg.text_cond_scale
        mineclip_config = MINECLIP_CONFIG
        mineclip_config['ckpt']['path'] = self.policy_cfg.mineclip_weights

        prior_info = PRIOR_INFO
        prior_info['model_path'] = self.policy_cfg.prior_weights
        self.prior = load_vae_model(prior_info, device=torch.device("cpu"))
        self.policy = make_agent(self.policy_cfg.in_model, self.policy_cfg.in_weights, cond_scale=self.cond_scale, device=torch.device("cpu")).policy

        self.mineclip = load_mineclip.load(mineclip_config, device=torch.device("cpu"))

        self.prompt_embeds = {}

    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        return recursive_tensor_op(lambda x : x.to(self.device), self.policy.initial_state(2))

    @torch.no_grad()
    def get_embed(self, prompt: str):
        if prompt not in self.prompt_embeds:
            self.prompt_embeds[prompt] = get_prior_embed(prompt, self.mineclip, self.prior, self.device)
        return torch.from_numpy(self.prompt_embeds[prompt]).to(self.device)

    # @property
    def action_head(self):
        return self.policy.pi_head

    @torch.cuda.amp.autocast()
    def forward(self, obs: Dict[str, torch.Tensor], state_in: Optional[List[torch.Tensor]], first: Optional[torch.Tensor], **kwargs):
        assert first is None or not first.any()
        B, T = obs['img'].shape[:2]
        assert B == 1 and T == 1
        first = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        print(obs['text'][0][0])
        obs['mineclip_embed'] = self.get_embed(obs['text'][0][0])

        obs = {"img": obs["img"], "mineclip_embed": obs["mineclip_embed"]}
        obs = tree_map(lambda x: torch.cat([x, x], dim=0), obs)
        obs["mineclip_embed"][1] = torch.zeros_like(obs["mineclip_embed"][1])
        first = torch.cat([first, first], dim=0)


        state_in = recursive_tensor_op(lambda x: x.squeeze(0), state_in)

        (pd, vpred, _), state_out = self.policy(obs=obs, first=first, state_in=state_in)
        if self.cond_scale is not None:
            pd = tree_map(lambda x: (((1 + self.cond_scale) * x[0]) - (self.cond_scale * x[1])).unsqueeze(0), pd)

        state_out = recursive_tensor_op(lambda x: x.unsqueeze(0), state_out)

        return {"pi_logits": pd, "vpred": vpred}, state_out, {}

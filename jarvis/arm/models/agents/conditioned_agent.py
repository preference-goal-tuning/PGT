import functools
import pickle
import torch
import re
import av
import cv2
import numpy as np
import typing

from rich import print
from rich.console import Console
from typing import Union, Dict, Optional, List, Tuple, Any
from omegaconf import DictConfig, ListConfig
import gymnasium.spaces.dict as dict_spaces

from jarvis.arm.utils import fit_img_space
from jarvis.arm.utils.vpt_lib.action_head import ActionHead
from jarvis.arm.models.policys import make_policy, load_policy_cfg
from jarvis.arm.models.agents.base_agent import BaseAgent

def convert_to_normal(obj):
    if isinstance(obj, DictConfig) or isinstance(obj, Dict):
        return {key: convert_to_normal(value) for key, value in obj.items()}
    elif isinstance(obj, ListConfig) or isinstance(obj, List):
        return [convert_to_normal(item) for item in obj]
    else:
        return obj

class ConditionedAgent(BaseAgent):
    
    def __init__(
        self, 
        state_space: Dict[str, Any] = {},
        action_space: Dict[str, Any] = {}, 
        policy_config: Union[DictConfig, str] = {}, 
        infer_env: str = 'minecraft', 
        weights_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        '''
        Arguments:
            state_space: Dict[str, Any]
                The state space of the environment, excluding the `image (RGB) observation`. 
            action_space: Dict[str, Any]
                The action space of the environment. 
            policy_config: Union[DictConfig, str]
                The configuration of the policy. 
            infer_env: str
                The environment where the policy rolls out.
        '''
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.policy_config = policy_config
        self.infer_env = infer_env

        if isinstance(self.policy_config, str):
            self.policy_config = load_policy_cfg(self.policy_config)
        
        self.policy, self.policy_building_info = make_policy(
            policy_cfg=self.policy_config, 
            state_space=self.state_space,
            action_space=self.action_space, 
            weights_dict=weights_dict, 
        )

        self.timesteps = self.policy_config['policy_kwargs']['timesteps']
        self.resolution = self.policy_config['policy_kwargs']['backbone_kwargs']['img_shape'][:2]
        
        self.cached_init_states = {}
        self.cached_first = {}
    
    @functools.lru_cache(maxsize=None)
    def direct_read_latent(self, given_latent_file) -> torch.Tensor:
        with open(given_latent_file, 'rb') as f:
            latent = pickle.load(f)
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).to(self.device)
        elif isinstance(latent, torch.Tensor):
            latent = latent.to(self.device)
        else:
            raise TypeError('Given latent should be np.ndarray or torch.Tensor, got {}'.format(type(latent)))
        return latent

    def wrapped_forward(self, 
                        obs: Dict[str, Any], 
                        state_in: Optional[List[torch.Tensor]],
                        first: Optional[torch.Tensor] = None, 
                        latent: Optional[torch.Tensor] = None,
                        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        '''Wrap state and first arguments if not specified. '''
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                B, T = v.shape[:2]
                break

        state_in = self.initial_state(B) if state_in is None else state_in
        
        if latent is None and self.policy.is_conditioned() and 'obs_conf' in obs:
            # if condition vector does not exist, compute condition from `obs_conf`
            obs_conf = obs['obs_conf'].copy()
            obs_conf['text'] = obs['text'].copy()
            given_latent_file = obs_conf.get('given_latent', None)
            while isinstance(given_latent_file, list):
                given_latent_file = given_latent_file[0]
            if isinstance(given_latent_file, str) and given_latent_file.endswith('.pkl'):
                latent = self.direct_read_latent(given_latent_file=given_latent_file)
            else:
                latent = self.load_input_condition(obs_conf=obs_conf, resolution=self.resolution, obs=obs)
        
        if first is None:
            first = self.cached_first.get((B, T), torch.tensor([[False]], device=self.device).repeat(B, T))
            self.cached_first[(B, T)] = first
        
        return self.policy.forward(
            obs=obs, 
            first=first, 
            state_in=state_in, 
            ice_latent=latent, 
            **kwargs
        )
    
    @functools.lru_cache(maxsize=None)
    def encode_video(
        self, ref_video: str, resolution: Tuple[int, int], avg: bool = False
    ) -> torch.Tensor:
        frames = []
        with av.open(ref_video, "r") as container:
            for fid, frame in enumerate(container.decode(video=0)):
                resized_frame = cv2.resize(
                    frame.to_ndarray(format="rgb24"), 
                    (resolution[0], resolution[1]), interpolation=cv2.INTER_LINEAR
                )
                frames.append(resized_frame)

        if avg:
            num_segs = len(frames) // self.timesteps
            segment = torch.stack(
                [torch.from_numpy(frame).to(self.device) for frame in frames[:num_segs*self.timesteps]], dim=0
            ).unsqueeze(0).reshape(num_segs, self.timesteps, resolution[0], resolution[1], 3)
        else:
            segment = torch.stack(
                [torch.from_numpy(frame).to(self.device) for frame in frames[:self.timesteps]], dim=0
            ).unsqueeze(0)
        conditions = self.policy.encode_condition(
            use_encoder='uni-encoder',
            obs={'img': segment}, 
            texts=[''],
            condition_info={
                'unique_token': 0, 
                'use_modal_name': ['episode_all_frames'],
            }
        )
        ce_latent = conditions.mean(0)
        return ce_latent

    @functools.lru_cache(maxsize=None)
    def encode_text(self, text: str, resolution: Tuple[int, int]) -> torch.Tensor:
        conditions = self.policy.encode_condition(
            use_encoder='uni-encoder',
            obs={'img': torch.zeros((1, 1, *resolution, 3), device="cuda")}, 
            texts=[text],
            condition_info={
                'unique_token': 3, 
                'use_modal_name': ['episode_text'], 
            }
        )
        ce_latent = conditions.mean(0)
        return ce_latent

    def load_input_condition(self, obs_conf: Dict, resolution: Tuple[int, int], obs: Dict) -> torch.Tensor:
        '''
        Load the input condition specified by the obs_conf. 
        For now, we only support video as input condition. 
        State sequence as the condition will be supported later!
        '''
        ins_type = obs_conf['ins_type'][0][0]
        if ins_type == 'video':
            assert 'ref_video' in obs_conf, 'ref_video should be specified in obs_conf. '
            num = len(obs_conf['ref_video'])
            ice_latent = []
            for i in range(num):
                ref_video = obs_conf['ref_video'][i][0]
                ce_latent = self.encode_video(ref_video=ref_video, resolution=tuple(resolution))
                ice_latent.append(ce_latent)
        elif ins_type == 'text':
            assert 'text' in obs_conf, 'text should be specified in obs_conf. '
            num = len(obs_conf['text'])
            ice_latent = []
            for i in range(num):
                ce_latent = self.encode_text(text=obs_conf['text'][i][0], resolution=tuple(resolution))
                ice_latent.append(ce_latent)
        return torch.stack(ice_latent, dim=0)
    
    def action_head(self) -> ActionHead:
        return self.policy.action_head(self.infer_env)

    @property
    def value_head(self) -> torch.nn.Module:
        return self.policy.value_head
    
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.policy.initial_state(1)]
        else:
            if batch_size not in self.cached_init_states:
                self.cached_init_states[batch_size] = [t.to(self.device) for t in self.policy.initial_state(batch_size)]
            return self.cached_init_states[batch_size]

    def forward(self, 
                obs: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]],
                first: Optional[torch.Tensor] = None,
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        forward_result, state_out, latents = self.wrapped_forward(obs=obs, state_in=state_in, first=first, **kwargs)
        return forward_result, state_out, latents

    
    
    def latent_from_encoder(self, use_encoder: str, obs: Optional[Dict] = None, **kwargs) -> torch.Tensor:
        '''
        Sample latent condition from the learned prior. 
        Arguments:
            use_encoder: str, the name of prior encoder. 
            obs: use_encoder may require the observation sequence [B, T, H, W, C]. 
            kwargs: includes other conditions such as returns and text.
        Returns:
            latent: torch tensor [B, S, D] indicates the latent condition.
        '''
        return self.policy.net.encode_condition(obs=obs, use_encoder=use_encoder, **kwargs)
    
    def decorate_obs(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        '''
        Convert the observation from environment to the format that the policy can understand.
        For example, numpy.array[84, 84] -> { torch.Tensor[128, 128, 3] cuda }
        Arguments:
            obs: np.ndarray, the observation from the environment.
        Returns:
            obs: Dict[str, torch.Tensor], the observation policy received. 
        '''
        assert isinstance(obs, np.ndarray), 'The observation should be a numpy array. '
        if len(obs.shape) > 1:
            # image type (such as minecraft and ataris)
            fit_img = fit_img_space([obs], resolution=self.resolution, to_torch=True, device=self.device)[0]
            return {'img': fit_img}
        else:
            # state type (such as mujoco and meta-world)
            return {f'{self.infer_env}_state': torch.from_numpy(obs).to(torch.float32).to(self.device)}
    

    @staticmethod
    def from_pretrained(ckpt_path: str, **kwargs):
        '''
        Load the agent from the checkpoint.
        '''
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        assert 'state_dict' in checkpoint, 'The checkpoint does not contain the state_dict. '
        build_helper_hparams = convert_to_normal(checkpoint['hyper_parameters']['build_helper_hparams'])
        build_helper_hparams.update(kwargs)
        instance = ConditionedAgent(**build_helper_hparams, weights_dict=checkpoint['state_dict'])
        return instance

def test_minecraft():
    from jarvis.stark_tech.env_interface import MinecraftWrapper
    e = MinecraftWrapper('diverses/collect_grass', prev_action_obs=True)
    o, _ = e.reset()
    action_space = {
        'minecraft': e.action_space, 
    }
    agent = ConditionedAgent(obs_space=e.observation_space, action_space=action_space, policy_config='groot_eff_1x')
    action, state = agent.get_action(o, input_shape="*")
    o, _, _, _, _ = e.step(action)
    action, state = agent.get_action(o, state_in=state, input_shape="*")


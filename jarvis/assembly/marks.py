import os
import hydra
from hydra import compose, initialize
from pathlib import Path
from typing import (
    Dict, List, Tuple, Union, Callable, 
    Sequence, Mapping, Any, Optional
)
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from copy import deepcopy

import av
import cv2
import torch
import numpy as np

from torch.cuda.amp import autocast as autocast
import jarvis
from jarvis.arm.models import MixedAgent
from jarvis.gallary.craft_script import CraftScript, SmeltScript
from jarvis.stark_tech.env_interface import MinecraftWrapper

class MarkBase:
    
    def __init__(self, **kwargs):
        pass
        
    def reset(self):
        raise NotImplementedError
    
    def do(self):
        raise NotImplementedError
    
    def record_step(self):
        record_frames = getattr(self, 'record_frames', [])
        record_infos = getattr(self, 'record_infos', [])
        self.record_frames = record_frames + [self.info['pov']]
        self.record_infos = record_infos + [self.info]
    
    def make_traj_video(self):
        if getattr(self, 'record_frames', None) is None:
            return
        container = av.open("mark_test.mp4", mode='w', format='mp4')
        stream = container.add_stream('h264', rate=20)
        stream.width = 640 
        stream.height = 360
        stream.pix_fmt = 'yuv420p'
        for frame in self.record_frames:
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

class GROOT(MarkBase):
    
    def __init__(
        self, 
        policy_configs: Dict, 
        env: Optional[MinecraftWrapper] = None, 
        **kwargs, 
    ) -> None:
        super().__init__(**kwargs)
        self.env = env
        self.agent = MixedAgent(
            policy_configs=policy_configs,
            action_space=env.get_action_space(),
            obs_space=env.get_obs_space()
        ).cuda()
        self.agent.eval()
        
    
    def reset(self):
        self.terminated = self.truncated = False
        self.state = self.agent.initial_state()
        noop_action = self.env.noop_action()
        self.obs = self.env.step(noop_action)[0]
        self.time_step = 0
        self.episode_reward = 0
    
    @torch.no_grad()
    def compute_action(self, obs):
        action, self.state = self.agent.get_action(
            obs, self.state, first=None, input_shape='*'
        )
        return action 
    
    def step(self, obs_conf: Dict):
        self.obs['obs_conf'] = obs_conf # set task condition
        action = self.compute_action(self.obs)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.time_step += 1
        self.episode_reward += reward
        return self.obs, reward, self.terminated, self.truncated, self.info
    
    def do(
        self,
        obs_conf: Dict,
        timeout: int = 500,
        target_reward: float = 1.,
        monitor_fn: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[bool, Dict]:
        '''
        obs_conf: specify the reference video, video mask, and action logit scale for each sub-policy. 
        timeout: maximum time step for the agent to finish the task.
        target_reward: if the target reward is reached, return with True, otherwise False
        monitor_fn: callback function to monitor the agent's achievements.
        '''
        self.reset()
        bar = tqdm(total=timeout)
        while (
            not self.terminated 
            and not self.truncated
            and self.time_step < timeout
        ):
            bar.update(1)
            self.step(obs_conf)
            if monitor_fn is not None:
                monitor_result = monitor_fn(self.info)
                if monitor_result[0]:
                    return monitor_result
            self.record_step()
            if self.episode_reward >= target_reward:
                return True, {}
        return False, {}

class MarkCrafter(MarkBase):
    
    def __init__(
        self,
        env: Optional[MinecraftWrapper] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.craft_script = CraftScript(env=env)
        self.smelt_script = SmeltScript(env=env)

    def reset(self): 
        self.craft_script.reset(fake_reset=True)
        self.smelt_script.reset(fake_reset=True)
        self.record_frames = []
        self.record_infos = []

    def do(self, condition: str = '', *args, **kwargs):
        # TODO: handle the corner case that the inventory is open at the beginning
        # need a signal: close inventory(when crafting/smelting for the last time)
        if condition == 'craft':
            result, error_message = self.craft_script.crafting(*args, **kwargs)
            self.record_frames += self.craft_script.outframes
            self.record_infos += self.craft_script.outinfos
        elif condition == 'smelt':
            result, error_message = self.smelt_script.smelting(*args, **kwargs)
            self.record_frames += self.smelt_script.outframes
            self.record_infos += self.smelt_script.outinfos
        else:
            raise ValueError("Condition must be `craft` or `smelt`. ")
        return result, error_message
         


if __name__ == '__main__':
    
    env = MinecraftWrapper('diverses/collect_wood')
    env.reset()
    
    policy_configs = {
        'A': 'groot_1x_minecraft_cnn'
    }
    
    mark = GROOT(policy_configs=policy_configs, env=env)
    mark.reset()
    
    obs_conf = {
        "A": {
            'task': 'build: dig three fill one',
            'ref_video': os.path.join(os.environ['JARVISBASE_TRAJS'], 'diverses/build_dig3fill1/human/0.mp4'),
            'ins_type': "video"
        }
    }
    
    mark.do(obs_conf, timeout=200, target_reward=2000)
    mark.make_traj_video()
    
    # '''
    # After doing a series of crafting/smelting, you need to close inventory/crafting_table/furnace
    # '''
    # # crafting

    # # smelting

    # # crafting


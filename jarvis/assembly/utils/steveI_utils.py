from argparse import Action
from copy import deepcopy
import pickle
from typing import Iterable, List
import numpy as np
from omegaconf import DictConfig
import torch
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.steveI.agents import SteveITextAgent
from jarvis.assembly.utils.video_utils import resize_varray, video2np
import os

project_root = '' # TODO: Set the project root

policy_config = DictConfig({
    'in_model': os.path.join(project_root, 'checkpoints', 'steve', 'vpt2x.model'),
    'in_weights': os.path.join(project_root, 'checkpoints', 'steve', 'steve1.weights'),
    'prior_weights': os.path.join(project_root, 'checkpoints', 'steve', 'steve1_prior.pt'),
    'mineclip_weights': os.path.join(project_root, 'checkpoints', 'mineclip', 'attn.pth'),
    'text_cond_scale': 6.0,
    'visual_cond_scale': 7.0
})
action_space = MinecraftWrapper.get_action_space()
observation_space = MinecraftWrapper.get_obs_space(width=128, height=128)
device = torch.device('cuda')

agent = SteveITextAgent(policy_config=policy_config, action_space=action_space, obs_space=observation_space).to(device)
agent.eval()
for param in agent.parameters():
    param.requires_grad = False


def to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        return obj
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj).to(device)
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, list):
        obj = [to_device(x, device) for x in obj]
        return obj
    if isinstance(obj, tuple):
        obj = tuple([to_device(x, device) for x in obj])
        return obj


def npify(dic):
    for k in dic:
        dic[k] = np.array(dic[k])
        if len(dic[k].shape) == 0 or dic[k].shape[0] != 1:
            dic[k] = dic[k][np.newaxis, ...]
    return dic


def format_actions(actions_list:List[Action]):
    batch_size = len(actions_list)
    minecraft_action = {
        'buttons': torch.empty(size=(batch_size, 128, 1), device=device, dtype=torch.int64),
        'camera': torch.empty(size=(batch_size, 128, 1), device=device, dtype=torch.int64)
    }
    for i, actions in enumerate(actions_list):
        if isinstance(actions, (tuple, list)):
            assert isinstance(actions[0], dict)
            if 'attack' in actions[0]:
                for j, action in enumerate(actions):
                    env_action = npify(deepcopy(action))
                    agent_action = MinecraftWrapper.env_action_to_agent(env_action)
                    minecraft_action['buttons'][i][j][0] = agent_action['buttons'].item()
                    minecraft_action['camera'][i][j][0] = agent_action['camera'].item()
        elif isinstance(actions, dict):
            if 'buttons' in actions:
                if isinstance(actions['buttons'], torch.Tensor):
                    minecraft_action['buttons'][i] = actions['buttons'].view(128, 1).to(device)
                    minecraft_action['camera'][i] = actions['camera'].view(128, 1).to(device)
                elif isinstance(actions['buttons'], np.ndarray):
                    minecraft_action['buttons'][i] = torch.from_numpy(actions['buttons']).view(128, 1).to(device)
                    minecraft_action['camera'][i] = torch.from_numpy(actions['camera']).view(128, 1).to(device)
    return minecraft_action


def get_loss(video, actions, given_latent=None):
    varray = video2np(video)
    varray = resize_varray(varray, 128, 128)
    vtensor = torch.from_numpy(varray[np.newaxis, ...]).to(device)
    
    if isinstance(given_latent, torch.Tensor):
        given_latent = given_latent.to(device)
    elif isinstance(given_latent, np.ndarray):
        given_latent = torch.from_numpy(given_latent).to(device)
    
    T = vtensor.shape[1]
    expanded = given_latent.view(-1).expand(size=(1, T, -1))
    obs = {'img': vtensor, 'mineclip_embed': expanded}
    first = torch.from_numpy(np.array([[True] + [False] * (T - 1)], dtype=bool).reshape(1, T)).to(device)
    agent_state = to_device(agent.policy.initial_state(1), device)
    (pi_h, v_h), state_out = agent.policy.net.forward(ob=obs, state_in=agent_state, context={"first": first})
    pi_logits = agent.policy.pi_head.forward(pi_h, mask=None)
    with open(actions, 'rb') as f:
        actions = pickle.load(f)
    actions = format_actions(actions_list=[actions])
    log_prob = agent.policy.get_logprob_of_action(pi_logits, actions)
    
    return torch.abs(log_prob)

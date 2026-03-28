import argparse
from copy import deepcopy
import pickle
import cv2
from tqdm import tqdm
from jarvis.assembly.utils.video_utils import video2np, np2video, np2image, resize_varray
import os
import torch
from jarvis.arm.models.agents.conditioned_agent import ConditionedAgent
from typing import Tuple, TypeVar, Union, Optional, Any, List, Dict
import numpy as np
from jarvis.stark_tech.env_interface import MinecraftWrapper


device = torch.device('cuda')
action_space = MinecraftWrapper.get_action_space()
observation_space = MinecraftWrapper.get_obs_space(width=128, height=128)

project_root = '' # TODO: Set the project root
pretrained_ckpt = os.path.join(project_root, 'checkpoints', 'groot', 'weight-epoch=8-step=80000.ckpt')

agent4latent:ConditionedAgent = ConditionedAgent.from_pretrained(pretrained_ckpt, action_space={'minecraft': action_space}, obs_space=observation_space, infer_env='minecraft').to(device)
agent4latent.eval()
for p in agent4latent.parameters():
    p.requires_grad = False

Action = TypeVar('Action', tuple, list, dict)
RGBFrames = TypeVar('RGBFrames', np.ndarray, torch.Tensor)

def rotate(elem, bias:int):
    if isinstance(elem, (tuple, list)):
        return elem[bias:] + elem[:bias]
    if isinstance(elem, np.ndarray):
        return np.concatenate((elem[bias:], elem[:bias]))
    if isinstance(elem, torch.Tensor):
        return torch.concatenate((elem[bias:], elem[:bias]))
    if isinstance(elem, dict):
        return {k: rotate(elem[k], bias) for k in elem}
    else:
        raise NotImplementedError


@torch.no_grad()
def get_latent(video:Union[str, RGBFrames], mode='posterior'):
    
    post = mode.startswith('post')
    
    if isinstance(video, str):
        varray = video2np(video)[:128]
    elif isinstance(video, np.ndarray):
        varray = video[:128]
    elif isinstance(video, torch.Tensor):
        varray = video[:128].numpy()
    else:
        raise ValueError(f'Unsupported video type {type(video)}!')
    
    reshaped = np.empty(shape=(128, 128, 128, 3), dtype=np.uint8)
    for fid in range(128):
        reshaped[fid] = cv2.resize(varray[fid], dsize=(128, 128))
    
    segment = torch.from_numpy(reshaped).unsqueeze(0).to(device)
    condition_info = {
        'unique_token': 0, 
        'use_modal_name': ['episode_all_frames' if post else 'episode_start_frame'],
    }
    ob_latent = agent4latent.policy.net.encode_observations({'img': segment})
    encoder_module = agent4latent.policy.net.encoders['uni-encoder']
    multimodal_inputs = dict(vision_feats=ob_latent['latent'], ob_is_padding=ob_latent['is_padding'], texts=[''])
    encoder_results = encoder_module(multimodal_inputs=multimodal_inputs, condition_info=condition_info)
    return encoder_results['space_result']


def fixed_info(batch_size):
    text = ['raw' for _ in range(batch_size)]
    mask = torch.ones(size=(batch_size, 128), device=device, dtype=torch.uint8)
    env = ['minecraft' for _ in range(batch_size)]
    condition_info = {
        'condition_name': ['start' for _ in range(batch_size)], 
        'use_modal_vector': torch.tensor([[0., 1., 0., 0.] for _ in range(batch_size)], device=device, dtype=torch.float32), 
        'informative': torch.zeros(size=(batch_size,), device=device, dtype=torch.bool), 
        'text_flag': torch.zeros(size=(batch_size,), device=device, dtype=torch.bool),
    }
    text_tokens = {
        'input_ids': torch.tensor([[ 101, 6315,  102] for _ in range(batch_size)], device=device, dtype=torch.int64), 
        'token_type_ids': torch.zeros(size=(batch_size, 3), device=device, dtype=torch.int64), 
        'attention_mask': torch.ones(size=(batch_size, 3), device=device, dtype=torch.int64), 
        'text_mask': torch.ones(size=(batch_size,), device=device, dtype=torch.int64)
    }
    return text, mask, env, condition_info, text_tokens


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


def generate_model_input(video_list:List[RGBFrames], actions_list:List[str], bias=0):
    batch_size = len(actions_list)
    text, mask, env, condition_info, text_tokens = fixed_info(batch_size)
    img = torch.empty(size=(batch_size, 128, 128, 128, 3), dtype=torch.uint8, device=device)
    for i in range(batch_size):
        img[i] = torch.from_numpy(np.concatenate((
                resize_varray(video_list[i], h=128, w=128)[bias:], 
                np.zeros((bias, 128, 128, 3), dtype=np.uint8)
            ))
        ).to(device)
    actions_list_read = []
    for actions in actions_list:
        if isinstance(actions, str):
            with open(actions, 'rb') as f:
                ori_actions = pickle.load(f)
                actions_list_read.append(rotate(ori_actions, bias))
        else:
            actions_list_read.append(rotate(actions, bias))
    minecraft_action = format_actions(actions_list_read)
    return {
        'text': text,
        'minecraft_action': minecraft_action, 
        'img': img,
        'mask': mask,
        'env': env,
        'condition_info': condition_info,
        'text_tokens': text_tokens
    }


def get_cross_loss(agent:ConditionedAgent, encoder_video, decoder_video_list, decoder_actions_list, given_latent=None, bias=0):
    """
    Returns:
        loss_buttons, loss_camera, action-wise logits buttons, action-wise logits camera
    """
    cross_latent = get_latent(video2np(encoder_video)) if given_latent is None else given_latent
    if decoder_video_list is not None and not isinstance(decoder_video_list, list):
        decoder_video_list = [decoder_video_list]
    if decoder_actions_list is not None and not isinstance(decoder_actions_list, list):
        decoder_actions_list = [decoder_actions_list]
    if decoder_video_list is None:
        assert len(decoder_actions_list) == 1
        decoder_vnp = [video2np(encoder_video)]
    else:
        assert len(decoder_video_list) == len(decoder_actions_list)
        decoder_vnp = [video2np(decoder_video) for decoder_video in decoder_video_list]
    input = generate_model_input(video_list=decoder_vnp, actions_list=decoder_actions_list, bias=bias)
    forward_result = agent.forward(obs=input, state_in=None, first=None, aux_latent=cross_latent)[0]    
    pi_head = forward_result['minecraft']['pi_head']
    pi_logits = pi_head(forward_result['minecraft']['pi_latent'])
    log_prob = pi_head.logprob(input['minecraft_action'], pi_logits, return_dict=True)
    camera_mask = (input['minecraft_action']['buttons'].int() % 2 != 0).float().squeeze(-1)
    logp_buttons = torch.abs(log_prob['buttons'])
    logp_camera  = torch.abs(log_prob['camera'] * camera_mask)
    return logp_buttons, logp_camera, pi_logits['buttons'], pi_logits['camera']

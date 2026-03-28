import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from typing import (
    List, Dict, Optional, Callable, Any, Tuple
)
from rich import print
from rich.console import Console

from jarvis.arm.utils.vpt_lib.misc import transpose
from jarvis.arm.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from jarvis.arm.models.backbones import build_backbone, build_state_backbones
from jarvis.arm.models.encoders.action import Action
from jarvis.arm.models.encoders.body import Image, Encoder, build_encoders
from jarvis.arm.models.fusions import build_condition_fusion_layer

def filter_loss(input: Dict):
    return dict(**{k: v for k, v in input.items() if 'loss' in k})

class GrootPolicy(nn.Module):
    
    def __init__(  
        self,
        state_space: Dict[str, Any] = {},
        action_space: Dict[str, Any] = {},
        hidsize: int = 512,
        init_norm_kwargs: Dict = {},
        # Below are TransformerXL's arguments
        attention_mask_style: str = "clipped_causal",
        attention_heads: int = 8,
        attention_memory_size: int = 1024,
        use_pointwise_layer: bool = True,
        pointwise_ratio: int = 4,
        pointwise_use_activation: bool = False,
        n_recurrence_layers: int = 4,
        recurrence_is_residual: bool = True,
        timesteps: int = 128,
        word_dropout: float = 0.0,
        # Below are custimized arguments
        backbone_kwargs: Dict = {},
        encoders_kwargs: Optional[List[Dict]] = None, 
        action_fusion: bool = False,
        condition_fusion_kwargs: Optional[Dict] = None,
        latent_space_kwargs: Dict = {},
        bc_only = False,
        **unused_kwargs,
    ):
        super().__init__()

        self.hidsize = hidsize
        self.timesteps = timesteps
        self.resolution = backbone_kwargs.get("resolution", None)
        self.bc_only = bc_only
        
        # Prepare necessary parameters. (required when load vanilla VPT)
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        # Build visual backbone module. 
        backbone_kwargs = {**backbone_kwargs, **unused_kwargs}
        backbone_kwargs['hidsize'] = hidsize
        backbone_kwargs['init_norm_kwargs'] = init_norm_kwargs
        backbone_kwargs['dense_init_norm_kwargs'] = self.dense_init_norm_kwargs
        backbone_results = build_backbone(**backbone_kwargs)
        self.img_process = backbone_results['obsprocessing']
        
        # Build state backbone modules. 
        self.state_backbones = build_state_backbones(state_space=state_space, hidsize=hidsize) if len(state_space) > 0 else None
        
        # Build latent space transformation layer.
        if (latent_hidsize := latent_space_kwargs.get('latent_hidsize', None)) is not None:
            self.transform = nn.Linear(latent_hidsize, hidsize)
        
        # Build posterior/prior encoders. 
        self.encoders = build_encoders(
            hidsize, encoders_kwargs=encoders_kwargs, latent_space_kwargs=latent_space_kwargs
        ) if encoders_kwargs else dict()
        
        # Build condition fusion layer. 
        self.condition_fusion_layer = Image(hidsize, **condition_fusion_kwargs) if condition_fusion_kwargs else None
        
        # Build action encoder layer. 
        self.action_fusion = Action(hidsize, action_space=action_space) if action_fusion else None
        
        # Build TransformerXL layer (decoder as policy). 
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type="transformer",
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
            inject_condition=self.is_conditioned(),
            word_dropout=word_dropout,
        ) 

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)
        self.cached_init_states = {}

    def extract_vision_feats(self, img: torch.Tensor) -> torch.Tensor:
        """
        Get vision features from the raw input image. 
        :params img: raw input image tensor with shape (B, T, H, W， 3) 
        :returns: vision features with shape (B, T, C) or (B, T, C, Hx, Wx)
                  depending on the backbone architecture (w/ pooling or w/o). 
        """
        if self.resolution is not None:
            if isinstance(self.resolution, int):
                assert img.shape[-3:-1] == (self.resolution, self.resolution), \
                    f"the observation resolution {img.shape[-3:-1]} does not match the agent resolution {self.resolution}"
            else:
                raise NotImplementedError
        B, T = img.shape[:2]
        x = self.img_process(img)
        vision_feats = x.reshape((B, T) + x.shape[2:])
        return vision_feats

    def extract_state_feats(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get state features from the raw input states. 
        :params states: raw input state tensor with shape (B, T, Cin)
        :returns: state features with shape (B, T, C)
        """
        return self.state_backbones(states)

    def encode_observations(self, observations: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode vision and state observations to one latent vector. 
        :params observations: a dictionary of observations, including 'img' and/or 'state'
        :returns: a latent vector with shape (B, T, C), and a is_padding tensor with shape (B, T)
        """
        if 'mask' not in observations:
            is_padding = torch.zeros(observations['img'].shape[:2], dtype=torch.bool, device=self.device)
        else:
            is_padding = (observations['mask'] == 0)
        
        feats = []
        # extract vision feats if image exists
        if 'img' in observations:
            vision_feat = self.extract_vision_feats(observations['img'])
            feats.append(vision_feat)
        
        # extract state feats if state vector exists
        states = {key.split('_state')[0]: val for key, val in observations.items() if '_state' in key}
        if len(states) > 0:
            state_feat = self.extract_state_feats(states)
            feats.append(state_feat)
        # fuse feats with a simple sum
        feat = torch.stack(feats, dim=0).sum(dim=0)
        return {
            "latent": feat, 
            "is_padding": is_padding
        }
    
    def encode_condition(
        self, obs: Optional[Dict[str, Any]] = None, use_encoder: str = 'uni-encoder', condition_info: Dict = None, **kwargs
    ):
        """
        Called from outside to encode condition latent from posterior or prior encoder. 
        :params obs: a dictionary of observations, including 'img' and/or 'state'
        :params infer: whether to infer the latent space or not
        """
        # extract vision feats 
        ob_latent = self.encode_observations(obs) if obs is not None else None
        # pickup the encoder module
        encoder_module = self.encoders[use_encoder]
        # kwargs also include necessary infomation such as 'texts' and 'returns'
        multimodal_inputs = dict(vision_feats=ob_latent['latent'], ob_is_padding=ob_latent['is_padding'], **kwargs)
        encoder_results = encoder_module(multimodal_inputs=multimodal_inputs, condition_info=condition_info)
        mu = encoder_results['space_result']['mu']
        log_var = encoder_results['space_result']['log_var']
        return encoder_results['space_result']['z']
    
    def encode_dists(self, ob_latent: Dict[str, torch.Tensor], obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Compute the posterior and prior distributions for the latent space. 
        :params obs: a dictionary of observations, including 'texts' and 'returns'. 
        :params ob_latent: the latent vector of the observations. 
        """
        assert 'uni-encoder' in self.encoders, "The encoder named 'uni-encoder' is required. "
        uni_encoder = self.encoders['uni-encoder']
        multimodal_inputs = dict(
            vision_feats=ob_latent['latent'], 
            ob_is_padding=ob_latent['is_padding'], 
            returns=obs.get('returns', None), 
            texts=obs.get('text', None), 
            text_tokens=obs.get('text_tokens', None)
        )
        # encode video to get distributions
        condition_info = obs['condition_info'].copy()
        condition_info['use_modal_vector'] = torch.zeros_like(condition_info['use_modal_vector'])
        condition_info['use_modal_vector'][..., 0] = 1.0
        video_results = uni_encoder(multimodal_inputs=multimodal_inputs, condition_info=condition_info)
        # encode others to get distributions
        condition_info = obs['condition_info'].copy()
        other_results = uni_encoder(multimodal_inputs=multimodal_inputs, condition_info=condition_info)
        # dispatch them into posterior and prior distributions
        informative = obs['condition_info']['informative'][:, None, None] # (B, 1, 1), torch.bool
        latents = {'prior': {'space_result': {}}, 'posterior': {'space_result': {}}}
        for key in video_results['space_result'].keys():
            latents['posterior']['space_result'][key] = torch.where(informative, other_results['space_result'][key], video_results['space_result'][key])
            latents['prior']['space_result'][key] = torch.where(informative, video_results['space_result'][key], other_results['space_result'][key])
        ce_latent = latents['posterior']['space_result']['z']
        
        
        if self.bc_only:
            ce_latent = latents['posterior']['space_result']['mu'] # sample from more informative one
        return ce_latent, latents, {}
    
    
    def decode(self, obs: Dict[str, Any], ob_latent: Dict[str, torch.Tensor], state_in: Dict, context: Dict, ce_latent: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Given the vision features, transformer state and condition latent, decode the final output for predicting actions.
        :params ob_latent: vision features with shape (B, T, C) or (B, T, C, Hx, Wx)
        :params state_in: the input state for transformerXL layer
        :params context: the context information for transformerXL layer
        :params ce_latent: the condition latent to control the decoder (policy network)
        """
        # transform ce_latent for conditioning
        if ce_latent is not None:
            ce_latent = self.transform(ce_latent)
        # use condition latent to pool the vision features
        if self.condition_fusion_layer is not None:
            x = self.condition_fusion_layer(vision_feats=ob_latent['latent'], ob_is_padding=ob_latent['is_padding'], latent=ce_latent)['tokens']
        else:
            x = ob_latent['latent']
        
        # fuse action embeddings
        if self.action_fusion is not None:
            prev_actions = {key.split('_prev_action')[0]: val for key, val in obs.items() if '_prev_action' in key}
            y = self.action_fusion(actions=prev_actions)
            x = x + y
        
        # pass into TransformerXL layer
        x, state_out = self.recurrent_layer(x, context["first"], state_in, ce_latent=ce_latent)
        
        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        return pi_latent, vf_latent, state_out
    
    def forward(self, obs: Dict, state_in: Dict, context: Dict, ice_latent: Optional[torch.Tensor] = None, **kwargs) -> Dict:
        # 0. get aux latent for test
        aux_latent = kwargs.pop('aux_latent') if kwargs.get('aux_latent', None) is not None else None
        
        # 1. extract observation features (once a forward)
        ob_latent = self.encode_observations(obs)
        
        # 2. encode distributions for latent space or use the given latent
        ce_latent, latents, internal_loss = ice_latent, {}, {}
        if ce_latent is None and self.is_conditioned():
            ce_latent, latents, internal_loss = self.encode_dists(ob_latent, obs)
            
        if aux_latent is not None:
            if isinstance(aux_latent, dict):
                ce_latent = aux_latent['mu']
            else:
                ce_latent = aux_latent

        # 3. decode features for action prediction
        pi_latent, vf_latent, state_out = self.decode(obs, ob_latent, state_in, context, ce_latent)
        
        # 4. return intermediate latents for decision making and other auxiliary tasks. 
        latents.update({
            "ob_latent": ob_latent,   # features of encoded images or states
            "pi_latent": pi_latent,   # features to predicting actions
            "vf_latent": vf_latent,   # features to predicting values
        })
        
        return latents, state_out, internal_loss

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            if batchsize not in self.cached_init_states:
                self.cached_init_states[batchsize] = self.recurrent_layer.initial_state(batchsize)
            return self.cached_init_states[batchsize]
        else:
            return None

    def output_latent_size(self):
        return self.hidsize

    def is_conditioned(self):
        return len(self.encoders) > 0
    
    @property
    def device(self):
        return next(self.parameters()).device
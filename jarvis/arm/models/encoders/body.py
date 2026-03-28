import random
import numpy as np
import torch
from torch import nn
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange, repeat
from copy import deepcopy

from jarvis.arm.models.encoders.vision import Image
from jarvis.arm.models.encoders.language import Language
from jarvis.arm.models.encoders.scalar import Scalar
from jarvis.arm.models.encoders.spaces import build_latent_space
from jarvis.arm.models.utils import MultimodalTransformer, FeedForward

def build_multimodal_backbone(hidsize:int, **kwargs):
    modal = kwargs.pop('modal')
    modal_mappings = {
        'image': Image, 
        'language': Language, 
        'scalar': Scalar, 
    }
    return modal_mappings[modal](hidsize=hidsize, **kwargs)

class ImageSlicerOperator(nn.Module):
    
    def __init__(self, source: str, target: str, operation: str):
        super().__init__()
        self.source = source
        self.target = target
        self.operation = operation
    
    def forward(self, features: Dict[str, Any]) -> Dict[str, Any]:
        expr_token = f"features[self.source]['tokens'][:, {self.operation}, ...]"
        expr_padding = f"features[self.source]['is_padding'][:, {self.operation}, ...]"
        return {self.target: {
            'tokens': eval(expr_token),
            'is_padding': eval(expr_padding)
        }}

def build_light_feature_operator(operator_name: str, **kwargs):
    if operator_name == 'image_slicer':
        return ImageSlicerOperator(**kwargs)
    else:
        raise NotImplementedError(f"Can not found {operator_name = }")

class Encoder(nn.Module):
    """
    (vision, text, returns) -> multimodal backbone -> multimodal transformer -> feedforwards -> latent space -> (z, mu, sigma). 
    """
    def __init__(
        self, 
        hidsize: int, 
        alias: str, 
        modal_list: List[str], 
        multimodal_backbone_kwargs: List[Dict] = [],
        light_feature_operator_kwargs: List[Dict] = [],
        multimodal_transformer_kwargs: Dict = {},
        latent_space_kwargs: Dict = {}, 
    ) -> None:
        super().__init__()
        self.alias = alias
        self.hidsize = hidsize
        self.modal_list = modal_list
        # build multimodal backbone
        self.multimodal_backbones = nn.ModuleDict()
        for kwargs in multimodal_backbone_kwargs:
            modal_name = kwargs.pop('modal_name')
            self.multimodal_backbones[modal_name] = build_multimodal_backbone(hidsize, **kwargs)
        # build light feature operators
        self.light_feature_operators = nn.ModuleList()
        for kwargs in light_feature_operator_kwargs:
            operator_name = kwargs.pop('operator_name')
            self.light_feature_operators.append( build_light_feature_operator(operator_name, **kwargs) )
        # build multimodal transformer
        self.multimodal_transformer = MultimodalTransformer(hidsize, **multimodal_transformer_kwargs)
        self.feedforwards = nn.ModuleList([FeedForward(hidsize, mult=2) for _ in range(2)])
        # build latent variable space
        self.latent_space = build_latent_space(hidsize, **latent_space_kwargs)
        # condition tokens 
        self.condition_tokens = nn.Embedding(10, hidsize)
    
    def make_condition_info(self, unique_token: int, use_modal_name: List[str], device: str = "cuda", **kwargs) -> Dict:
        """
        Make condition_info dict for evaluation. 
        :params unique_token: indicate which condition to be used. 
        :params use_modal_name: List[str], indicate which modal to be used. 
        :returns: condition_info for func make_condition. 
        """
        if self.alias == 'posterior':
            assert 'episode_all_frames' in use_modal_name, "episode_all_frames must be used in posterior."
        use_modal_vector = torch.zeros(len(self.modal_list), device=device)
        for idx, modal_name in enumerate(self.modal_list):
            if modal_name not in use_modal_name:
                continue
            use_modal_vector[idx] = 1
        return {
            'use_modal_vector': use_modal_vector[None],
        }
    
    def make_condition(self, feats_dict: Dict[str, torch.Tensor], condition_info: Optional[Dict] = None):
        """
        Make all the required features to build conditions. 
        :params feats_dict: Dict[str, Any], keys are the names of modal inputs.
        """
        feats_dict = feats_dict.copy()

        if condition_info is None:
            return feats_dict

        if 'use_modal_vector' not in condition_info:
            assert 'use_modal_name' in condition_info, "if use_modal_vector is not provided, use_modal_name must be provided."
            condition_info = self.make_condition_info(**condition_info, device=self.device)
        
        if self.alias == 'posterior': #! use one shared posterior and disable unique_token parameter
            return {'episode_all_frames': feats_dict['episode_all_frames']}
        
        use_modal_vector = condition_info['use_modal_vector']
        for idx, modal_name in enumerate(self.modal_list):
            mask = repeat(use_modal_vector[:, idx], 'b -> b m', m=feats_dict[modal_name]['tokens'].shape[1]).to(torch.bool)
            feats_dict[modal_name]['is_padding'] = feats_dict[modal_name]['is_padding'] | (~mask) # type: ignore
            if feats_dict[modal_name]['is_padding'].all():
                feats_dict.pop(modal_name)

        
        return feats_dict
    
    def forward(self, multimodal_inputs: Dict[str, Any], condition_info: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Fuse multimodal inputs and generate latent space representation. 
        :params multimodal_inputs: Dict[str, Any], keys are the names of modal inputs. 
        :params condition_kwargs: Optional[Dict], keys are the names of condition inputs.
        """
        feats_dict = {
            modal_name: self.multimodal_backbones[modal_name](**multimodal_inputs) \
                for modal_name in self.modal_list if modal_name in self.multimodal_backbones
        }
        for operator in self.light_feature_operators:
            feats_dict.update(operator(feats_dict))
        x = self.make_condition(feats_dict, condition_info) # B, M, C
        x = self.multimodal_transformer(x) # B, C
        for ffn in self.feedforwards:
            x = ffn(x) + x
        x = rearrange(x, 'b c -> b 1 c')
        space_result = self.latent_space(x)
        return {'space_result': space_result}
    
    @property
    def device(self):
        return next(self.parameters()).device

def build_encoders( hidsize: int, encoders_kwargs: List, **kwargs ) -> Dict[str, nn.Module]:
    result = {}
    for private_kwargs in encoders_kwargs['private_kwargs']:
        this_kwargs = deepcopy(encoders_kwargs['public_kwargs'])
        this_kwargs.update(deepcopy(private_kwargs))
        result[this_kwargs['alias']] = Encoder(hidsize, **this_kwargs, **kwargs)
    return nn.ModuleDict(result)

if __name__ == '__main__':
    """
    debug the Encoder class.
    """
    encoder_kwargs = {
        'hidsize': 256,
        'alias': 'hybrid encoder',
        'multimodal_backbone_kwargs': [
            {
                'modal': 'image', 
                'select': ':', 
                'squeeze': True,
            }, 
            {
                'modal': 'language', 
            }, 
            {
                'modal': 'scalar'
            }
        ], 
        'multimodal_transformer_kwargs': {}, 
        'latent_space_kwargs': {
            'type': 'VAE', 
            'latent_hidsize': 512, 
        }
    }
    
    encoder = Encoder(**encoder_kwargs).to("cuda")
    
    B, T, C, H, W = 4, 16, 256, 7, 7
    vision_feats = torch.randn(B, T, C, H, W).to("cuda")
    texts = ['A', 'B2', 'Cat', 'Dog']
    returns = torch.randn(B, ).to("cuda")
    
    print(f"{vision_feats.shape = }")
    print(f"{returns.shape = }")
    print(f"{texts = }")
    
    output = encoder({
        'vision_feats': vision_feats, 
        'texts': texts, 
        'returns': returns
    })
    
    print(f"{output['space_result']['z'].shape = }")
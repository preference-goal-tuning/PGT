import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from functools import partial
from typing import Dict, Optional, Union, List, Any, Tuple
from einops import rearrange

from jarvis.arm.models.backbones.state import DictStateEncoder
from jarvis.arm.models.backbones.clipv import CustomCLIPv
from jarvis.arm.models.backbones.swin_transformer import CustomSwinTransformer
from jarvis.arm.models.backbones.impala import ImgObsProcess
from jarvis.arm.models.backbones.resnet import CustomResNet
from jarvis.arm.models.backbones.efficient import CustomEfficientNet
from jarvis.arm.models.backbones.nfnet import CustomNFNet


    


    
    
    

def build_backbone(name: str = 'IMPALA', **kwargs) -> Dict:
    
    result_modules = {}
    if name == 'IMPALA':
        first_conv_norm = False
        impala_kwargs = kwargs.get('impala_kwargs', {})
        init_norm_kwargs = kwargs.get('init_norm_kwargs', {})
        dense_init_norm_kwargs = kwargs.get('dense_init_norm_kwargs', {})
        result_modules['obsprocessing'] = ImgObsProcess(
            cnn_outsize=kwargs.get('cnn_outsize', 256),
            output_size=kwargs['hidsize'],
            inshape=kwargs['img_shape'],
            chans=tuple(int(kwargs['impala_width'] * c) for c in kwargs['impala_chans']),
            nblock=2,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs, 
        )
        if (film_kwargs := impala_kwargs.get('film_kwargs')) is not None:
            result_modules['uncond_embedding'] = nn.Embedding(1, film_kwargs["cond_dim"])
        
    elif name == 'CLIPv':
        model = CustomCLIPv(
            out_dim=kwargs['hidsize'],
            **kwargs,
        )
        result_modules['obsprocessing'] = model

    elif name == 'SWIN':
        model = CustomSwinTransformer(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['obsprocessing'] = model

    elif name == 'NFNET':
        model = CustomNFNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['obsprocessing'] = model
    
    elif name == 'EFFICIENTNET':
        model = CustomEfficientNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
        result_modules['obsprocessing'] = model
        
    elif name == 'RESNET':
        result_modules['obsprocessing'] = CustomResNet(
            out_dim=kwargs['hidsize'],
            **kwargs, 
        )
    return result_modules

def build_state_backbones(
    state_space: Dict[str, Any], hidsize: int, **kwargs
):
    return DictStateEncoder(state_space=state_space, hidsize=hidsize)

if __name__ == '__main__':
    pass 
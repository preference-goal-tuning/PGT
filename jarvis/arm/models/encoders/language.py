import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Union, List, Any
from einops import rearrange, repeat
from jarvis.arm.utils.transformers import GPTConfig, GPT
from jarvis.arm.models.utils import FeedForward, ModalInput

class AttentionPooling(nn.Module):
    
    def __init__(self, hidsize: int, **transformer_kwargs) -> None:
        super().__init__()
        gpt_config = GPTConfig(
            bias = True,
            is_causal = False,
            is_ordered = True,
            n_embd = hidsize,
            ** transformer_kwargs
        )
        self.transformer = GPT(gpt_config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidsize))
    
    def forward(self, vision_feats: torch.Tensor, latent: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        :params vision_feats: (B, S, C), where S is the spatial dimension. 
        :params latent: (B, 1, C), the condition latent vector, optional to be used. 
        :return: (B, 1, C), the fused vision features.
        """
        assert len(vision_feats.shape) == 3 and vision_feats.shape[1] > 1, "Spatial dimension is required."
        if latent is not None:
            x = torch.cat([latent, vision_feats], dim=1)
        else:
            x = vision_feats
        x = torch.cat([repeat(self.cls_token, '1 1 c -> b 1 c', b=x.shape[0]), x], dim=1)
        x = self.transformer(x)[:, 0, :]
        x = rearrange(x, 'b c -> b 1 c')
        return x

class Language(nn.Module):
    
    def __init__(
        self, 
        hidsize: int, 
        model_version: str = 'bert-base-uncased', 
        full_sequence: bool = True, 
        freeze: bool = True, 
        **kwargs
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.language = AutoModel.from_pretrained(self.model_version)
        self.updim = nn.Linear(self.language.config.hidden_size, hidsize)
        self.feedforwards = nn.ModuleList([
            FeedForward(hidsize, mult=2) for _ in range(2)
        ])
        
        self.full_sequence = full_sequence
        if not self.full_sequence:
            self.aggregate_layer = AttentionPooling(hidsize=hidsize, **kwargs)
        
        if freeze:
            self.freeze_language_model()

    def freeze_language_model(self):
        for param in self.language.parameters():
            param.requires_grad = False
    
    def encode_text_tokens(self, texts: Union[str, List[str]]):
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        assert isinstance(texts, list) and isinstance(texts[0], str)
        text_tokens = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.language.device)
        text_tokens['text_mask'] = torch.tensor([text != '' for text in texts]).float().to(self.language.device)
        return text_tokens

    def forward(self, texts: List[str], text_tokens: Optional[Dict] = None, **kwargs) -> ModalInput:
        '''
        :params text: List[str], the input text, useless during training, but useful during inference. 
        :params text_tokens: Optional[Dict], the tokenized text, useful during training. 
        :return text_feats: B, L, C where B is the batch size, L is the sequence length. 
        '''
        if text_tokens is None:
            text_tokens = self.encode_text_tokens(texts)
        else:
            text_tokens = text_tokens.copy()
        prior_mask = text_tokens.pop('text_mask') #! useful!!!
        is_padding = (text_tokens['attention_mask'] == 0)
        is_padding[prior_mask == 0] = 1
        x = self.language(**text_tokens).last_hidden_state
        x = self.updim(x)
        for ffn in self.feedforwards:
            x = ffn(x) + x
        
        if not self.full_sequence:
            x = self.aggregate_layer(x)
            is_padding = is_padding[:, :1]
        
        return {
            "tokens": x,
            "is_padding": is_padding
        }


if __name__ == '__main__':
    texts = ['put the red block on the green block', 'put the green block on the red block']
    language_module = Language(
        hidsize=1024, 
        full_sequence=False, 
        n_head=8, 
        n_layer=2, 
    ).cuda()
    text_feats = language_module(texts=texts)
    import ipdb; ipdb.set_trace()
    
    
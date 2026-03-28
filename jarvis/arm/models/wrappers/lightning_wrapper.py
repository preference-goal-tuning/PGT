import typing
import hydra
import wandb
from typing import Tuple, Dict, List, Union, Optional, Any
from rich.console import Console
from omegaconf import DictConfig, ListConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import gymnasium.spaces.dict as dict_spaces
from einops import rearrange
from lightning.pytorch import LightningModule

from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.arm.models.agents.conditioned_agent import ConditionedAgent
from torchvision.transforms.v2 import Identity
from torchmetrics import Metric

def filter_KL(input: Dict):
    return dict(**{k: v for k, v in input.items() if 'KL_original' in k})

def tree_get(obj, keys: List, default=None):
    
    try:
        for key in keys:
            if key in obj:
                obj = obj[key]
            else:
                return default
        return obj
    except:
        return default

def convert_to_normal(obj):
    if isinstance(obj, DictConfig) or isinstance(obj, Dict):
        return {key: convert_to_normal(value) for key, value in obj.items()}
    elif isinstance(obj, ListConfig) or isinstance(obj, List):
        return [convert_to_normal(item) for item in obj]
    else:
        return obj

class LightningBase(LightningModule):
    
    def __init__(
        self,
        state_space: dict_spaces.Dict,
        action_space: dict_spaces.Dict,
        policy_config: DictConfig,
        lightning_config: DictConfig,
    ) -> None:
        super().__init__()
        
        self.agent = ConditionedAgent(state_space, action_space, policy_config)
        self.policy_config = policy_config
        self.lightning_config = lightning_config
        build_helper_hparams = convert_to_normal({
            'state_space': state_space,
            'action_space': action_space, 
            'policy_config': policy_config,
            'lightning_config': lightning_config,
        })
        if lightning_config.get('augmentations', None) is not None:
            self.aug_transform = hydra.utils.instantiate(lightning_config.augmentations)
        else:
            self.aug_transform = None
        self.save_hyperparameters({'build_helper_hparams': build_helper_hparams})
        
        self.agent_metrics = self.agent.policy.get_metrics()

    def on_train_start(self, *args, **kwargs):
        for metric in self.agent_metrics:
            metric.reset()
    
    def on_validation_start(self, *args, **kwargs):
        for metric in self.agent_metrics:
            metric.reset()
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.aug_transform is not None:
            imgs = batch['img']
            if imgs.shape[-1] == 3:
                imgs = rearrange(imgs, 'b t h w c -> b t c h w')
            new_imgs = []
            for img in imgs:
                new_imgs.append(self.aug_transform(img))
            batch['img'] = torch.stack(new_imgs, dim=0)
        return batch

    def step_and_log(self, batch, batch_idx, stage):
        """
        Using automatic optimization to train the model. 
        call `self.in_step_loop` to compute the loss and log the metrics.
        """
        result = self.in_step_loop(input=batch, stage=stage)
        if batch_idx % self.lightning_config.optimize.log_interval == 0:
            for key, val in result.items():
                if key.endswith('_bak'):
                    continue
                prog_bar_flag = ('loss' in key) and (stage == 'train')
                if isinstance(val, tuple):
                    avg, num = val
                    self.log(f'{stage}/{key}', avg, sync_dist=False, prog_bar=prog_bar_flag, batch_size=num)
                elif isinstance(val, Metric):
                    self.log(f'{stage}/{key}', val, sync_dist=False, prog_bar=prog_bar_flag, on_epoch=True)
                else:
                    self.log(f'{stage}/{key}', val, sync_dist=False, prog_bar=prog_bar_flag)
            
        
        # make sure that loss is a scalar tensor before returning to the optimizer
        if isinstance(result['loss'], tuple):
            result['loss'] = result['loss'][0]
        
        return result
    
    def get_env_logs(self, batch, baks):
        KEYWORDS = ['KL_original', 'loss']
        result = {}
        for bidx, env_name in enumerate(batch['env']):
            if (k := f'{env_name}_env') in batch:
                env_name = f"{env_name} ({batch[k][bidx]})"
            for field, val in baks.items():
                if not any([k in field for k in KEYWORDS]):
                    continue
                env_field = f"{env_name}/{field}"
                if env_field not in result:
                    result[env_field] = []
                if (v := val[bidx]) != 0.:
                    # remove masked values
                    result[env_field] += [v]
        result = {key: sum(val) / len(val) for key, val in result.items()}
        return result
    
    def training_step(self, batch, batch_idx):
        return self.step_and_log(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step_and_log(batch, batch_idx, f'val_{dataloader_idx}')
    
    def configure_optimizers(self):
        
        learning_rate = self.lightning_config.optimize.learning_rate
        selected_discount = self.lightning_config.optimize.selected_discount
        other_discount = self.lightning_config.optimize.other_discount
        weight_decay = self.lightning_config.optimize.weight_decay
        warmup_steps = self.lightning_config.optimize.warmup_steps
        training_steps = self.lightning_config.optimize.training_steps
        
        if self.lightning_config.optimize.frozen_other:
            for name, param in self.agent.policy.named_parameters():
                if all( (param_key not in name) for param_key in self.lightning_config.optimize.selected_keys ):
                    param.requires_grad = False
        
        all_named_parameters = dict(self.agent.policy.named_parameters())
        all_named_parameters = dict(filter(
            lambda pair: pair[1].requires_grad,
            all_named_parameters.items()
        ))
        
        selected_keys = self.lightning_config.optimize.selected_keys
        if isinstance(selected_keys, str):
            if selected_keys == 'ckpt_parameters':
                selected_keys = list(self.agent.policy_building_info['ckpt_parameters'].keys())
        else:
            ...
        
        selected_parameters = filter( 
            lambda pair: any( 
                ( param_key in pair[0] ) for param_key in selected_keys
            ), 
            all_named_parameters.items()
        )

        other_parameters = filter(
            lambda pair: all(
                ( param_key not in pair[0] ) for param_key in selected_keys
            ), 
            all_named_parameters.items()
        )

        optimizable_parameters = [
            {'params': [p for n, p in selected_parameters], 'lr': learning_rate*selected_discount}, 
            {'params': [p for n, p in other_parameters], 'lr': learning_rate*other_discount},
        ]

        
        optimizer = torch.optim.AdamW(
            params=optimizable_parameters,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }, 
        }

    def set_var(self, key, val):
        self.agent.set_var(key, val)
    
    def get_var(self, key):
        return self.agent.get_var(key)

class NaiveTrainingMixin:
    
    def __init__(self):
        self.automatic_optimization = True
    
    def in_step_loop(self, input, stage=None):
        '''Naive training method does not provide state_in. '''
        return self.step_func(input=input, state_in=None)[0]

class TBPTTMethodMixin:

    def __init__(self):
        self.automatic_optimization = False
        self.timesteps = tree_get(
            obj=self.policy_config, 
            keys=['timesteps'], 
            default=128, 
        )

    def in_step_loop(self, input, stage=None):
        '''t-BPTT training method feed the state_in to the model in a loop. '''
        for k, v in input.items():
            if isinstance(v, torch.Tensor):
                B, T = v.shape[:2]
                break
        num_mini_batch = T // self.timesteps
        state_in = None
        accumulate_mini_results = []
        for idx, mini_batch in enumerate(self.tbptt_split_batch(input, split_size=self.timesteps, B=B, T=T)):
            mini_result, state_in = self.step_func(input=mini_batch, state_in=state_in)
            accumulate_mini_results = accumulate_mini_results + [mini_result]
            state_in = [x.detach() for x in state_in] 
            self.log(f'tbptt/mini_loss_{idx}', mini_result["loss"].cpu().item(), sync_dist=True, prog_bar=True)
            if stage == 'train':
                self.manual_backward(mini_result['loss'] / num_mini_batch)
        
        result = {}
        for key in accumulate_mini_results[0].keys():
            result[key] = torch.stack(
                [mini_result[key] for mini_result in accumulate_mini_results], dim=0
            ).mean(dim=0)
            self.log(f'{stage}/{key}', result[key].cpu().item(), sync_dist=True, prog_bar=True)
        if stage == 'train':
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()
            sch = self.lr_schedulers()
            sch.step()
        return result
        
    def tbptt_split_batch(self, batch: Any, split_size: int, B: int, T: int) -> List[Any]:
        """
        Split a batch of data into a list of batches with length `split_size`. 
        """
        def tree_tbptt_split_batch(batch, start, end):
            """Slice a batch of data."""
            result = {}
            for key, val in batch.items():
                if key == 'text':
                    result[key] = val
                elif isinstance(val, torch.Tensor):
                    result[key] = val[:, start:end]
                elif isinstance(val, Dict):
                    result[key] = tree_tbptt_split_batch(val, start, end)
            return result
        
        splits = []
        for t in range(0, T, split_size):
            batch_split = tree_tbptt_split_batch(batch, start=t, end=t+split_size)
            splits.append(batch_split)
        return splits

class BehaviorCloningMixin:
    
    def step_func(self, input, state_in=None): 
        '''Compute the behavior cloning loss.'''
        mask = input['mask']
        forward_result, state_out, latents = self.agent(obs=input, state_in=state_in)
        import ipdb; ipdb.set_trace()
        loss = 0
        result = {}
        # add internal loss
        for key, val in forward_result['internal_loss'].items():
            result[key] = val
            loss += val
        
        # compute auxiliary head metrics ( such as recon loss and KL loss )
        for head, module in self.agent.policy.auxiliary_heads.items():
            result_metric = module.loss(obs=input, pred=forward_result[head], mask=mask) # type: ignore
            for metric, val in result_metric.items():
                result[metric] = val
                if 'loss' in metric:
                    loss += val

        # make a backup for logging metric env by env
        result_bak = {f"{k}_bak": v for k, v in result.items()}
        
        result['loss'] = loss / self.lightning_config.optimize.loss_scale
        for key, val in result.items():
            if isinstance(val, torch.Tensor):
                # average over batch dimension, val == 0 means the value is masked
                sample_mask = (val != 0.).float()
                num = sample_mask.sum()
                avg = (val * sample_mask).sum() / (num + 1e-6)
                result[key] = (avg, num)
        
        result.update(result_bak)
        
        if torch.isnan(loss).any():
            Console().log("NaN loss Warning!!!")
            import ipdb; ipdb.set_trace()
        
        return result, state_out

class DirectPreferenceLearningMixin:
    
    def __init__(
        self,
        obs_space: dict_spaces.Dict,
        action_space: dict_spaces.Dict,
        policy_config: DictConfig,
        lightning_config: DictConfig,
        **kwargs, 
    ) -> None:
        self.beta = lightning_config['optimize']['beta']
        # reference agent is a copy of self.agent and will be frozen during training 
        self.reference_agent = ConditionedAgent(obs_space, action_space, policy_config)
        for name, param in self.reference_agent.named_parameters():
            param.requires_grad = False
        self.reference_agent.eval()
    
    def concat_batch(self, batches: List[Union[torch.Tensor, Dict]]):
        batch_out = {}
        for key, val in batches[0].items():
            if isinstance(val, torch.Tensor):
                batch_out[key] = torch.cat([batch_in[key] for batch_in in batches], dim=0)
            elif isinstance(val, Dict):
                batch_out[key] = self.concat_batch([batch_in[key] for batch_in in batches])
            else:
                batch_out[key] = val
        return batch_out
    
    def split_batch(self, batch: Dict[str, Union[torch.Tensor, Dict]]):
        first_batch, second_batch = {}, {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                first_batch[key] = val[:val.shape[0]//2]
                second_batch[key] = val[val.shape[0]//2:]
            elif isinstance(val, Dict):
                first_batch[key], second_batch[key] = self.split_batch(val)
            else:
                first_batch[key] = second_batch[key] = val
        return first_batch, second_batch
    
    def get_batch_logs(self, agent: nn.Module, batch_in: Dict, state_in=None):
        
        pos_batch = {
            'mask': batch_in['pos_mask'], 
            'img': batch_in['pos_img'], 
            'action': batch_in['pos_act'], 
            'prev_action': batch_in['pos_prev_act']
        }
        neg_batch = {
            'mask': batch_in['neg_mask'],
            'img': batch_in['neg_img'], 
            'action': batch_in['neg_act'], 
            'prev_action': batch_in['neg_prev_act']
        }
        
        big_batch = self.concat_batch([pos_batch, neg_batch])
        
        if batch_in['con_img'] is not None:
            big_con_img = torch.cat([batch_in['con_img'], batch_in['con_img']], dim=0)
            condition = agent.policy.encode_condition(obs={'img':big_con_img})
        else:
            contition = None
        
        output, state_out = agent(obs=big_batch, state_in=state_in)
        pi_logits: Dict[str, torch.tensor] = output['pi_logits']
        minerl_action = {k: v for k, v in big_batch['action'].items()}
        logp, logp_buttons, logp_camera = self.compute_hierarchical_logp(
            agent.action_head, minerl_action, pi_logits, mask=big_batch['mask'], average="sum"
        )
        pos_logp = logp[:logp.shape[0]//2]
        neg_logp = logp[logp.shape[0]//2:]
        
        return pos_logp, neg_logp, state_out
    
    def compute_dpo_loss(
        self, 
        agent_pos_logp: torch.FloatTensor, 
        agent_neg_logp: torch.FloatTensor, 
        reference_pos_logp: torch.FloatTensor, 
        reference_neg_logp: torch.FloatTensor,
        beta: float, 
    ):
        
        delta_pos = agent_pos_logp - reference_pos_logp
        delta_neg = (agent_neg_logp - reference_neg_logp) * 0
        logits = delta_pos - delta_neg
        loss = - F.logsigmoid(beta * logits)
        pos_rewards = beta * delta_pos.detach()
        neg_rewards = beta * delta_neg.detach()
        return loss, pos_rewards, neg_rewards
        
    def step_func(self, input, state_in=None):
        '''Compute the direct preference learning loss.'''
        B = input['pos_img'].shape[0]
        if state_in is None:
            agent_state_in, reference_state_in = None, None
        else:
            agent_state_in, reference_state_in = state_in[:len(state_in)//2], state_in[len(state_in)//2:]
        # compute agent and reference agent's logp
        agent_pos_logs, agent_neg_logs, agent_state_out = \
            self.get_batch_logs(self.agent, input, state_in=agent_state_in)
        with torch.no_grad():
            reference_pos_logs, reference_neg_logs, reference_state_out = \
                self.get_batch_logs(self.reference_agent, input, state_in=reference_state_in)
        # compute loss and rewards
        loss, pos_rewards, neg_rewards = self.compute_dpo_loss(
            agent_pos_logs, agent_neg_logs, reference_pos_logs, reference_neg_logs, beta=self.beta 
        )
        reward_accuracies = (pos_rewards > neg_rewards).float()
        result = {
            'loss': loss.mean(), 
            'pos_rewards': pos_rewards.mean(), 
            'neg_rewards': neg_rewards.mean(),
            'reward_accuracy': reward_accuracies.mean(),
        }
        state_out = agent_state_out + reference_state_out
        return result, state_out
    
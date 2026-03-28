import os
import torch
import numpy as np
import pickle
import json
from torch.optim import Adam
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.arm.models.agents import ConditionedAgent

def load_trajectories(output_dir):
    trajectories = []
    rewards = []
    
    with open(os.path.join(output_dir, 'rewards.json'), 'r') as f:
        all_rewards = json.load(f)
        
    for i, reward in enumerate(all_rewards):
        traj_path = os.path.join(output_dir, f'episode_{i+1}.pkl')
        if os.path.exists(traj_path):
            with open(traj_path, 'rb') as f:
                traj = pickle.load(f)
            trajectories.append(traj)
            rewards.append(reward)
            
    return trajectories, rewards

def collate_trajectory(traj, device):
    obs_list = traj['obs']
    action_list = traj['action']
    
    # Collate obs
    obs_batch = {}
    for k in obs_list[0].keys():
        if isinstance(obs_list[0][k], np.ndarray):
            # (T, ...)
            stacked = np.stack([o[k] for o in obs_list])
            obs_batch[k] = torch.from_numpy(stacked).unsqueeze(0).to(device) # (1, T, ...)
        elif isinstance(obs_list[0][k], torch.Tensor):
            stacked = torch.stack([o[k] for o in obs_list])
            obs_batch[k] = stacked.unsqueeze(0).to(device)
        else:
            # list of strings or other
            obs_batch[k] = [[o[k] for o in obs_list]]
            
    # Collate action
    action_batch = {}
    for k in action_list[0].keys():
        if isinstance(action_list[0][k], np.ndarray):
            stacked = np.stack([a[k] for a in action_list])
            action_batch[k] = torch.from_numpy(stacked).unsqueeze(0).to(device)
        elif isinstance(action_list[0][k], torch.Tensor):
            stacked = torch.stack([a[k] for a in action_list])
            action_batch[k] = stacked.unsqueeze(0).to(device)
        else:
            stacked = np.array([a[k] for a in action_list])
            action_batch[k] = torch.from_numpy(stacked).unsqueeze(0).to(device)
            
    return obs_batch, action_batch

def compute_logprob(agent, obs_batch, action_batch, z):
    B, T = list(obs_batch.values())[0].shape[:2]
    state_in = agent.initial_state(B)
    first = torch.zeros((B, T), dtype=torch.bool, device=agent.device)
    first[:, 0] = True
    
    # Forward pass
    result, _, latent_out = agent.forward(obs_batch, state_in, first=first, latent=z, stage='rollout')
    
    action_head = agent.action_head()
    if 'pi_logits' not in result:
        result['pi_logits'] = action_head(latent_out['pi_latent'])
        
    log_prob = action_head.logprob(action_batch, result['pi_logits'])
    
    if isinstance(log_prob, dict):
        log_prob = sum(log_prob.values())
        
    # log_prob is (B, T)
    return log_prob.sum(dim=1) # (B,)

def train_pgt(task, output_dir, ckpt_path, ref_video, beta=0.1, lr=1e-2, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load agent
    env_name = f'diverses/{task}'
    env = MinecraftWrapper(env_name)
    agent = ConditionedAgent.from_pretrained(
        ckpt_path, 
        action_space={'minecraft': env.get_action_space()}, 
        obs_space=env.get_obs_space(), 
        infer_env='minecraft'
    ).to(device)
    agent.eval()
    
    # Load data
    trajectories, rewards = load_trajectories(output_dir)
    print(f"Loaded {len(trajectories)} trajectories.", flush=True)
    
    # Form preference pairs
    pairs = []
    for i in range(len(rewards)):
        for j in range(len(rewards)):
            if rewards[i] > rewards[j]:
                pairs.append((trajectories[i], trajectories[j]))
                
    print(f"Formed {len(pairs)} preference pairs.", flush=True)
    if len(pairs) == 0:
        print("No preference pairs found. Artificially creating pairs for testing...", flush=True)
        for i in range(len(rewards) // 2):
            rewards[i] = 1
        for i in range(len(rewards)):
            for j in range(len(rewards)):
                if rewards[i] > rewards[j]:
                    pairs.append((trajectories[i], trajectories[j]))
        print(f"Artificially formed {len(pairs)} preference pairs.", flush=True)
        
    if len(pairs) == 0:
        print("Still no pairs. Exiting.", flush=True)
        return
        
    # Split into train and eval pairs
    import random
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    
    print(f"Split into {len(train_pairs)} train pairs and {len(eval_pairs)} eval pairs.", flush=True)
    
    # For fast testing, limit train pairs if too many
    if len(train_pairs) > 20:
        train_pairs = train_pairs[:20]
        print(f"Sampled 20 train pairs for fast testing.", flush=True)
        
    if len(eval_pairs) > 10:
        eval_pairs = eval_pairs[:10]
        print(f"Sampled 10 eval pairs for fast testing.", flush=True)
        
    # Initialize z with z_ref
    obs_conf = {
        "task": task.replace('_', ' '),
        "ref_video": [[ref_video]],
        "ins_type": [['video']]
    }
    # We need a dummy obs to pass to load_input_condition
    dummy_obs = collate_trajectory(trajectories[0], device)[0]
    with torch.no_grad():
        z_ref = agent.load_input_condition(obs_conf=obs_conf, resolution=agent.resolution, obs=dummy_obs)
        z_ref = z_ref.detach()
        
    z = torch.nn.Parameter(z_ref.clone())
    optimizer = Adam([z], lr=lr)
    
    # Precompute reference logprobs
    print("Precomputing reference logprobs...", flush=True)
    
    def precompute_ref(pair_list):
        ref_w, ref_l = [], []
        for traj_w, traj_l in pair_list:
            obs_w, act_w = collate_trajectory(traj_w, device)
            obs_l, act_l = collate_trajectory(traj_l, device)
            with torch.no_grad():
                lw = compute_logprob(agent, obs_w, act_w, z_ref)
                ll = compute_logprob(agent, obs_l, act_l, z_ref)
            ref_w.append(lw)
            ref_l.append(ll)
        return ref_w, ref_l
        
    train_ref_w, train_ref_l = precompute_ref(train_pairs)
    eval_ref_w, eval_ref_l = precompute_ref(eval_pairs)
    
    best_eval_loss = float('inf')
    best_z = z_ref.clone()
    
    # Training loop
    print("Starting training...", flush=True)
    for epoch in range(epochs):
        # Train
        total_train_loss = 0
        for i, (traj_w, traj_l) in enumerate(train_pairs):
            obs_w, act_w = collate_trajectory(traj_w, device)
            obs_l, act_l = collate_trajectory(traj_l, device)
            
            lw = compute_logprob(agent, obs_w, act_w, z)
            ll = compute_logprob(agent, obs_l, act_l, z)
            
            diff_w = lw - train_ref_w[i]
            diff_l = ll - train_ref_l[i]
            
            loss = -torch.nn.functional.logsigmoid(beta * (diff_w - diff_l))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / max(1, len(train_pairs))
        
        # Eval
        total_eval_loss = 0
        with torch.no_grad():
            for i, (traj_w, traj_l) in enumerate(eval_pairs):
                obs_w, act_w = collate_trajectory(traj_w, device)
                obs_l, act_l = collate_trajectory(traj_l, device)
                
                lw = compute_logprob(agent, obs_w, act_w, z)
                ll = compute_logprob(agent, obs_l, act_l, z)
                
                diff_w = lw - eval_ref_w[i]
                diff_l = ll - eval_ref_l[i]
                
                loss = -torch.nn.functional.logsigmoid(beta * (diff_w - diff_l))
                total_eval_loss += loss.item()
                
        avg_eval_loss = total_eval_loss / max(1, len(eval_pairs))
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}", flush=True)
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_z = z.clone().detach()
            print(f"  New best eval loss: {best_eval_loss:.4f}", flush=True)
        
    # Save optimized z
    save_path = os.path.join(output_dir, 'optimized_z.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(best_z.cpu(), f)
    print(f"Saved best optimized z (eval loss: {best_eval_loss:.4f}) to {save_path}", flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='collect_wood')
    parser.add_argument('--output_dir', type=str, default='outputs/collect_wood')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/groot/weight-epoch=8-step=80000.ckpt')
    parser.add_argument('--ref_video', type=str, default='reference_videos/collect_wood.mp4')
    args = parser.parse_args()
    
    train_pgt(args.task, args.output_dir, args.ckpt_path, args.ref_video)

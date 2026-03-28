import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.arm.models.agents import ConditionedAgent
from jarvis.assembly.utils.video_utils import np2video
import cv2
import json
import argparse
import pickle

def run_worker(rank, args, episodes_to_run, start_episode_idx):
    # Set visible GPU for current process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    env_name = args.env_name
    env = MinecraftWrapper(env_name)
    env.reset()
    
    ckpt_path = args.ckpt_path
    agent = ConditionedAgent.from_pretrained(
        ckpt_path, 
        action_space={'minecraft': env.get_action_space()}, 
        obs_space=env.get_obs_space(), 
        infer_env='minecraft'
    ).cuda()
    agent.eval()
    
    task_name = env_name.split('/')[-1]
    obs_conf = {
        "task": task_name.replace('_', ' '),
        "ref_video": args.ref_video,
        "ins_type": "video"
    }
    if args.given_latent:
        obs_conf["given_latent"] = args.given_latent

    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    max_steps = args.max_steps
    
    print(f"[Worker {rank}] Starting {episodes_to_run} episodes of {env_name}...")
    
    rewards = []
    
    for i in range(episodes_to_run):
        global_episode_idx = start_episode_idx + i + 1
        obs, info = env.reset()
        state = agent.initial_state()
        
        total_reward = 0
        frames = []
        trajectory = {'obs': [], 'action': []}
        
        for step in range(max_steps):
            if 'img' in obs:
                frames.append(obs['img'].copy())
            elif 'pov' in obs:
                frames.append(obs['pov'].copy())
            
            # Save obs (without obs_conf to save space, we can add it back later)
            saved_obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items() if k != 'obs_conf'}
            trajectory['obs'].append(saved_obs)
            
            obs['obs_conf'] = obs_conf
            if 'text' not in obs:
                obs['text'] = ''
                
            with torch.no_grad():
                action, state = agent.get_action(obs, state, first=None, input_shape='*')
            
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            elif isinstance(action, dict):
                action_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in action.items()}
            else:
                action_np = action
                
            trajectory['action'].append(action_np)
                
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            
            if terminated or truncated:
                break
                
        if 'img' in obs:
            frames.append(obs['img'].copy())
        elif 'pov' in obs:
            frames.append(obs['pov'].copy())
            
        print(f"[Worker {rank}] Episode {i+1}/{episodes_to_run} (Global {global_episode_idx}) finished with reward {total_reward}")
        rewards.append(total_reward)
        
        if len(frames) > 0:
            video_path = os.path.join(output_dir, f'episode_{global_episode_idx}.mp4')
            video_array = np.stack(frames)
            np2video(video_array, video_array.shape[2], video_array.shape[1], video_path)
            
        traj_path = os.path.join(output_dir, f'episode_{global_episode_idx}.pkl')
        with open(traj_path, 'wb') as f:
            pickle.dump(trajectory, f)
        
        # Save individual reward for each worker
        with open(os.path.join(output_dir, f'rewards_worker_{rank}.json'), 'w') as f:
            json.dump(rewards, f)

def run():
    parser = argparse.ArgumentParser(description="Run GROOT model rollouts in Minecraft")
    parser.add_argument('--env_name', type=str, default='diverses/collect_wood', help='Environment config path (e.g., diverses/collect_wood)')
    parser.add_argument('--ref_video', type=str, default='reference_videos/collect_wood.mp4', help='Path to reference video')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=128, help='Maximum steps per episode')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/groot/weight-epoch=8-step=80000.ckpt', help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--given_latent', type=str, default=None, help='Path to optimized latent goal (.pkl)')
    args = parser.parse_args()

    num_episodes = args.num_episodes
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found! Falling back to 1 worker.")
        num_gpus = 1
        
    print(f"Found {num_gpus} GPUs. Splitting {num_episodes} episodes across them.")
    
    episodes_per_worker = num_episodes // num_gpus
    remainder = num_episodes % num_gpus
    
    processes = []
    start_idx = 0
    
    for rank in range(num_gpus):
        episodes_to_run = episodes_per_worker + (1 if rank < remainder else 0)
        p = mp.Process(target=run_worker, args=(rank, args, episodes_to_run, start_idx))
        p.start()
        processes.append(p)
        start_idx += episodes_to_run
        
    for p in processes:
        p.join()
        
    print("All workers finished. Merging rewards...")
    
    # Merge rewards from all workers
    task_name = args.env_name.split('/')[-1]
    output_dir = os.path.join(args.output_dir, task_name)
    all_rewards = []
    for rank in range(num_gpus):
        worker_reward_file = os.path.join(output_dir, f'rewards_worker_{rank}.json')
        if os.path.exists(worker_reward_file):
            with open(worker_reward_file, 'r') as f:
                all_rewards.extend(json.load(f))
                
    with open(os.path.join(output_dir, 'rewards.json'), 'w') as f:
        json.dump(all_rewards, f)
        
    print(f"Saved total {len(all_rewards)} rewards to {os.path.join(output_dir, 'rewards.json')}")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    run()

# Preference Goal Tuning (PGT)

This repository contains the environment, rollout scripts, and training pipeline for **Preference Goal Tuning (PGT)** on goal-conditioned policies (like GROOT) in Minecraft.

## Overview

Goal-conditioned policies enable decision-making models to execute diverse behaviors based on specified goals. We formulate post-training adaptation as a latent control problem, where the goal embedding serves as a continuous control variable to modulate the behavior of a frozen policy.

This repository provides the infrastructure to:
1. Run the GROOT foundation policy in Minecraft.
2. Generate trajectory data (videos, actions, and rewards) for preference learning.
3. Train and optimize the latent goal ($z$) using a DPO-like loss based on preference pairs.
4. Test the optimized latent goal in the environment.

## Installation

### Prerequisites
- Python 3.9+
- Java 8 (for Minecraft environment `MCP-Reborn`)
- Conda (recommended for environment management)

### Setup Environment
Ensure you have the required dependencies installed. You can use the provided `pyproject.toml` or install the necessary packages manually (e.g., `torch`, `gymnasium`, `av`, `opencv-python`, `hydra-core`, `omegaconf`, `einops`, `rich`, `torchmetrics`).

```bash
# Example using conda
conda create -n pgt python=3.10
conda activate pgt
pip install -r requirements.txt # Or install based on pyproject.toml
```

*Note: The Minecraft environment (`MCP-Reborn`) requires specific Java setup and Xvfb/vglrun for headless rendering. Please ensure `launchClient.sh` is configured correctly for your system.*

## Usage Pipeline

The PGT pipeline consists of three main steps: Data Generation, Training, and Testing.

### 1. Data Generation (Rollout)

To generate rollouts for a specific task using the original GROOT policy (this serves as both testing the original policy and generating data for PGT), use the `run.py` script. 

This script runs multiple episodes in parallel across available GPUs.

```bash
python run.py \
    --env_name diverses/collect_wood \
    --ref_video reference_videos/collect_wood.mp4 \
    --num_episodes 50 \
    --max_steps 128 \
    --output_dir outputs
```

**Arguments:**
- `--env_name`: The environment configuration path (e.g., `diverses/collect_wood`). Configurations are located in `jarvis/global_configs/envs/`.
- `--ref_video`: The path to the reference video for the task, used to extract the initial latent goal.
- `--num_episodes`: Total number of episodes to generate.
- `--max_steps`: Maximum number of steps per episode.
- `--output_dir`: Directory to save the generated trajectories.

**Outputs:**
The generated episodes will be saved in the `outputs/<task>/` directory:
- `.mp4` video files for each episode.
- `.pkl` trajectory files containing detailed observations and actions.
- `rewards.json` containing the cumulative reward for each episode.

### 2. PGT Training

Once the data is generated, run the PGT training script to optimize the latent goal based on the preference pairs formed from the generated trajectories.

```bash
python train_pgt.py \
    --task collect_wood \
    --output_dir outputs/collect_wood \
    --ref_video reference_videos/collect_wood.mp4 \
    --beta 0.1 \
    --lr 1e-2 \
    --epochs 50
```

**How it works:**
1. Loads the generated `.pkl` trajectories and `rewards.json`.
2. Forms preference pairs $(\tau_w, \tau_l)$ where $R(\tau_w) > R(\tau_l)$.
3. Splits the pairs into training and evaluation sets.
4. Optimizes the latent goal $z$ using Adam optimizer to maximize the likelihood of preferred trajectories and minimize the likelihood of non-preferred ones (DPO loss).
5. Selects the best $z$ based on the lowest evaluation loss.
6. Saves the optimized latent goal to `outputs/<task>/optimized_z.pkl`.

### 3. Testing with Optimized Latent Goal

Finally, you can test the policy using the newly optimized latent goal by passing the `--given_latent` argument to `run.py`:

```bash
python run.py \
    --env_name diverses/collect_wood \
    --ref_video reference_videos/collect_wood.mp4 \
    --num_episodes 20 \
    --max_steps 128 \
    --given_latent outputs/collect_wood/optimized_z.pkl \
    --output_dir outputs_test
```

Compare the average rewards in `outputs_test/<task>/rewards.json` with the original generation step to evaluate the performance improvement.

## Repository Structure

- `run.py`: Main script for running environment rollouts and data collection.
- `train_pgt.py`: Script for training and optimizing the latent goal using PGT.
- `checkpoints/`: Directory containing pretrained policy weights (e.g., GROOT).
- `reference_videos/`: Directory containing reference videos for various tasks.
- `jarvis/`: Core library containing environment wrappers, model architectures, and configurations.
  - `stark_tech/`: Minecraft environment interface (`MinecraftWrapper`) and backend (`MCP-Reborn`).
  - `arm/`: Model architectures (`ConditionedAgent`, `GrootPolicy`, etc.).
  - `global_configs/`: Environment and task YAML configurations.
  - `assets/`: Minecraft static assets, recipes, and spawn configurations.

## Acknowledgements

This codebase builds upon the infrastructure of MineRL, VPT, and STEVE-1.

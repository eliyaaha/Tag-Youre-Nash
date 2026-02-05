from pettingzoo.mpe import simple_tag_v3
from .RewardSharing import *
import supersuit as ss
import torch
import os
import multiprocessing
import random
import numpy as np

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across python, numpy, and pytorch.
    """
    # 1. Base Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # 4. PyTorch Deterministic Operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed set to: {seed}")

# Set up the device and cores
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CORES = max(1, multiprocessing.cpu_count() - 2)
os.environ["SDL_VIDEODRIVER"] = "dummy"  # For headless server execution

def create_env(alpha=1.0, num_cores = -1, max_cycles = 50):
    """
    Creates a vectorized environment for faster training.
    """
    # 1. Initialize the base environment configuration
    env = simple_tag_v3.parallel_env(
        num_good=1, 
        num_adversaries=3, 
        num_obstacles=2, 
        max_cycles=max_cycles, 
        continuous_actions=False,
        render_mode=None
    )
    
    possible_agents = env.possible_agents
    world_obj = env.unwrapped.world

    # 2. Apply the Reward Sharing Wrapper
    env = RewardSharingWrapper(env, alpha=alpha)
    
    # 3. Apply SuperSuit wrappers for SB3 compatibility
    env = ss.pad_observations_v0(env) # Ensure equal observation size
    env = ss.pad_action_space_v0(env) # Ensure equal action size
    env = ss.pettingzoo_env_to_vec_env_v1(env) # Convert to Vector Env

    if num_cores == -1:
        num_cores = NUM_CORES

    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs=1,  
        num_cpus=num_cores,      
        base_class="stable_baselines3"
    )
    return env, possible_agents, world_obj
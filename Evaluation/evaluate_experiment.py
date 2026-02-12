import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Modular imports from project structure
from Helpers.RewardSharing import *
from Helpers.Enviroment import create_env, DEVICE
from Helpers.Logger import ExperimentLogger

def process_step_metrics(env, rewards, agent_names, info):
    """
    Extracts high-level metrics and agent trajectories from a single environment step.
    
    Args:
        env: The vectorized environment (to access live world state).
        rewards: Numpy array of rewards from the vectorized environment.
        agent_names: List of agent names in the environment.
        info: Info dictionary from the environment step.
        
    Returns:
        tuple: (step_reward, is_capture, collisions, step_coords)
    """
    # 1. Calculate combined reward for all predators (adversaries)
    predator_indices = [i for i, a in enumerate(agent_names) if "adversary" in a]
    current_rewards = rewards[0] if len(rewards.shape) > 1 else rewards
    step_reward = np.sum(current_rewards[predator_indices])
    
    # 2. Identify if a capture occurred (based on reward threshold)
    is_capture = step_reward > 8
    
    # 3. Extract collision counts and coordinates from the live environment
    collisions = 0
    current_info = info[0] if isinstance(info, list) else info
    if isinstance(current_info, dict):
        for agent_data in current_info.values():
            if isinstance(agent_data, dict) and agent_data.get("collision", False):
                collisions += 1

    # 4. Extract XY Coordinates by unwrapping the vectorized environment
    step_coords = []
    
    try:
        # Unwrap through the SuperSuit/VectorEnv wrapper stack to reach the underlying world
        # Path: SB3VecEnvWrapper -> venv (ConcatVecEnv) -> vec_envs[0] (MarkovVectorEnv)
        #       -> par_env (aec_to_parallel_wrapper) -> aec_env (OrderEnforcingWrapper) -> world
        
        markov_env = env.venv.vec_envs[0]
        par_env = markov_env.par_env
        aec_to_par = par_env.aec_env  # This is actually the aec wrapper
        
        # Go back through aec_to_parallel_wrapper to get aec_env with world
        aec_env = aec_to_par
        while hasattr(aec_env, 'env') and not hasattr(aec_env, 'world'):
            aec_env = aec_env.env
        
        # Try to get world from aec_to_parallel_wrapper
        base_par_env = aec_to_par
        while hasattr(base_par_env, 'env'):
            base_par_env = base_par_env.env
        
        if hasattr(base_par_env, 'aec_env'):
            actual_aec = base_par_env.aec_env
            if hasattr(actual_aec, 'world'):
                world = actual_aec.world
                
                # Extract positions from all agents
                for agent in world.agents:
                    step_coords.append({
                        "agent name": agent.name,
                        "x coordinate": float(agent.state.p_pos[0]),
                        "y coordinate": float(agent.state.p_pos[1])
                    })
                
                # Infer collisions from live positions as fallback
                if collisions == 0:
                    predator_agents_objs = [a for a in world.agents if "adversary" in a.name]
                    for i in range(len(predator_agents_objs)):
                        for j in range(i + 1, len(predator_agents_objs)):
                            p1 = np.array(predator_agents_objs[i].state.p_pos)
                            p2 = np.array(predator_agents_objs[j].state.p_pos)
                            r1 = predator_agents_objs[i].size
                            r2 = predator_agents_objs[j].size
                            dist = np.linalg.norm(p1 - p2)
                            if dist <= r1 + r2:
                                collisions += 1
            else:
                # Fallback: return NaN if world not found
                step_coords = [{"agent name": name, "x coordinate": np.nan, "y coordinate": np.nan} 
                              for name in agent_names]
        else:
            # Fallback: return NaN if aec_env not found
            step_coords = [{"agent name": name, "x coordinate": np.nan, "y coordinate": np.nan} 
                          for name in agent_names]
    except Exception as e:
        # Fallback: return NaN on any unwrapping error
        print(f"Warning: Could not extract positions during step: {e}")
        step_coords = [{"agent name": name, "x coordinate": np.nan, "y coordinate": np.nan} 
                      for name in agent_names]
                
    return step_reward, is_capture, collisions, step_coords

def evaluate_model(alpha, episodes=50, max_cycles=50):
    """
    Main evaluation loop for a specific alpha model.
    Saves summary metrics and full XY trajectory data to CSV.
    """
    logger = ExperimentLogger(experiment_name="Evaluation", alpha=alpha)
    
    # Setup paths
    model_path = f"./models/predator_alpha_{alpha}"
    results_dir ="results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if model exists before proceeding
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # Create environment
    env, possible_agents, _ = create_env(alpha=alpha, max_cycles=max_cycles, eval=True)
    
    logger.info(f"Loading PPO model from {model_path}")
    model = PPO.load(model_path, device=DEVICE)
    
    # Data storage
    episode_summary_data = []
    trajectory_data = []

    logger.info(f"Starting Evaluation | Alpha: {alpha} | Episodes: {episodes}")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward, ep_captures, ep_collisions = 0, 0, 0
        first_capture_step = np.nan
        
        for step in range(1, max_cycles + 1):
            # Agent-environment interaction
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # Extract metrics using the modular processor
            step_reward, is_capture, collisions, step_coords = process_step_metrics(
                env, rewards, possible_agents, infos
            )
            
            # Update episode accumulation
            ep_reward += step_reward
            ep_collisions += collisions
            
            if is_capture:
                ep_captures += 1
                if np.isnan(first_capture_step):
                    first_capture_step = step
            
            # Store agent trajectories with metadata
            for coord in step_coords:
                coord.update({
                    "episode id": ep,
                    "cycle id": step
                })
                trajectory_data.append(coord)

        # Log progress to console/file
        logger.log_metrics(ep, ep_reward, ep_captures, ep_collisions)
        
        episode_summary_data.append({
            "episode": ep,
            "cumulative_reward": ep_reward,
            "total_captures": ep_captures,
            "total_collisions": ep_collisions
        })

    # --- Data Export ---
    # 1. Create a specific directory for this alpha (e.g., results/alpha_0.5/)
    alpha_results_dir = os.path.join(results_dir, f"alpha_{alpha}")
    os.makedirs(alpha_results_dir, exist_ok=True)
    
    # 2. Save Episode Summaries (One row per episode)
    summary_df = pd.DataFrame(episode_summary_data)
    summary_csv = os.path.join(alpha_results_dir, "summary.csv") 
    summary_df.to_csv(summary_csv, index=False)
    
    # 3. Save Trajectories (One row per agent per step)
    trajectory_df = pd.DataFrame(trajectory_data)
    column_order = ["agent name", "episode id", "cycle id", "x coordinate", "y coordinate"]
    trajectory_df = trajectory_df[column_order]
    
    trajectory_csv = os.path.join(alpha_results_dir, "trajectories.csv")
    trajectory_df.to_csv(trajectory_csv, index=False)
    
    logger.info(f"Evaluation finished for Alpha {alpha}")
    logger.info(f"📁 All files saved to: {alpha_results_dir}")
    logger.info(f"   - {summary_csv}")
    logger.info(f"   - {trajectory_csv}")

if __name__ == "__main__":
    # Standard CLI support for running evaluation independently
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value to evaluate")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--max_cycles", type=int, default=50, help="Max cycles per episode")
    
    args = parser.parse_args()
    evaluate_model(alpha=args.alpha, episodes=args.episodes, max_cycles=args.max_cycles)
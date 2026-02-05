import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from Helpers.RewardSharing import *
from Helpers.Enviroment import *
from Helpers.Logger import ExperimentLogger

def evaluate_model(alpha, episodes=50, max_cycles=50):
    """
    Evaluates a trained model for a specific alpha value and logs:
    1. Cumulative Predator Reward
    2. Steps to first capture (Time to capture)
    3. Total captures per episode
    4. Total collisions per episode
    """
    
    # Initialize the custom logger from our Helpers folder
    logger = ExperimentLogger(experiment_name="Evaluation", alpha=alpha)
    
    model_path = f"models/predator_alpha_{alpha}"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # Create the environment with the correct alpha and max_cycles
    # Ensure create_env in Helpers.Enviroment accepts these parameters
    env, possible_agents = create_env(alpha=0.0, max_cycles=max_cycles, num_cores=-1)
    
    logger.info(f"Loading PPO model from {model_path}")
    model = PPO.load(model_path, device=DEVICE)
    
    # List to store dictionaries of episode data for easy conversion to DataFrame
    raw_data = []

    logger.info(f"Starting Evaluation | Alpha: {alpha} | Episodes: {episodes} | Max Cycles: {max_cycles}")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0
        ep_captures = 0
        ep_collisions = 0
        first_capture_step = np.nan # Use NaN to represent "no capture occurred"
        
        for step in range(1, max_cycles + 1):
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # 1. Calculate Cumulative Predator Reward
            # Note: We sum rewards for all agents identified as "adversary"
            predator_rewards = [rewards[i] for i, a in enumerate(possible_agents) if "adversary" in a]
            step_reward = np.sum(predator_rewards)
            ep_reward += step_reward
            
            # 2. Track Captures (First and Total)
            # Based on project logic: reward > 8 indicates a successful tag/capture
            if step_reward > 8:
                ep_captures += 1
                if np.isnan(first_capture_step):
                    first_capture_step = step
            
            # 3. Track Collisions
            # Extract collision data from the info dictionary provided by the environment
            current_info = infos[0] if isinstance(infos, list) else infos
            if isinstance(current_info, dict):
                for agent_data in current_info.values():
                    if isinstance(agent_data, dict) and agent_data.get("collision", False):
                        ep_collisions += 1

        # Log episode summary using our custom Logger class
        logger.log_metrics(ep, ep_reward, ep_captures, ep_collisions)
        
        # Append all collected metrics for this episode
        raw_data.append({
            "episode": ep,
            "cumulative_reward": ep_reward,
            "first_capture_step": first_capture_step,
            "total_captures": ep_captures,
            "total_collisions": ep_collisions
        })

    # --- Data Export & Final Summary ---

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(raw_data)
    
    # Ensure data directory exists
    if not os.path.exists("results"):
        os.makedirs("results")
        
    csv_filename = f"results/alpha_{alpha}_results.csv"
    df.to_csv(csv_filename, index=False)
    
    # Final logging summary
    logger.info("-" * 30)
    logger.info(f"Evaluation finished for Alpha {alpha}")
    logger.info(f"Results saved to: {csv_filename}")
    logger.info(f"Average Reward: {df['cumulative_reward'].mean():.2f}")
    logger.info(f"Average Captures: {df['total_captures'].mean():.2f}")
    
    # Calculate mean only for episodes where a capture actually happened
    avg_first_cap = df['first_capture_step'].dropna().mean()
    logger.info(f"Average Steps to First Capture: {avg_first_cap:.2f}")
    logger.info("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a MARL model")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value of the model to evaluate")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("--max_cycles", type=int, default=50, help="Max steps per episode")
    
    args = parser.parse_args()

    evaluate_model(
        alpha=args.alpha, 
        episodes=args.episodes, 
        max_cycles=args.max_cycles
    )
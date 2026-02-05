import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from Helpers.RewardSharing import *
from Helpers.Enviroment import *
import pandas as pd

def evaluate_model(alpha, episodes=50, max_cycles=50):
    """
    Evaluates the model and logs raw data per episode.
    """
    model_path = f"models/predator_alpha_{alpha}"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Model {model_path}.zip not found.")
        return

    # Pass max_cycles to the environment creator
    env = create_env(alpha=0.0, max_cycles=max_cycles)
    pred_model = PPO.load(model_path, device=DEVICE)
    
    # Raw data storage for later analysis/graphing
    raw_rewards = []
    raw_first_capture_steps = []
    raw_total_captures = []
    raw_total_collisions = []
    
    print(f"\n🚀 Evaluating Predator Alpha {alpha} | Episodes: {episodes} | Max Cycles: {max_cycles}")

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0
        ep_captures = 0
        ep_collisions = 0
        first_capture_step = np.nan 
        
        for step in range(max_cycles):
            action, _ = pred_model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            
            # 1. Cumulative Reward
            predator_rewards = [rewards[i] for i, a in enumerate(env.possible_agents) if "adversary" in a]
            step_reward = np.sum(predator_rewards)
            ep_reward += step_reward
            
            # 2. Captures (First and Total)
            if step_reward > 8:
                ep_captures += 1
                if np.isnan(first_capture_step):
                    first_capture_step = step + 1 
            
            # 3. Collisions
            current_info = infos[0] if isinstance(infos, list) else infos
            if isinstance(current_info, dict):
                for agent_data in current_info.values():
                    if isinstance(agent_data, dict) and agent_data.get("collision", False):
                        ep_collisions += 1

        # Save episode results to lists
        raw_rewards.append(ep_reward)
        raw_first_capture_steps.append(first_capture_step)
        raw_total_captures.append(ep_captures)
        raw_total_collisions.append(ep_collisions)
        
        if (ep + 1) % 10 == 0:
            print(f"   Processed {ep + 1}/{episodes} episodes...")

    # Export to CSV
    df = pd.DataFrame({
        "episode": range(1, episodes + 1),
        "cumulative_reward": raw_rewards,
        "first_capture_step": raw_first_capture_steps,
        "total_captures": raw_total_captures,
        "total_collisions": raw_total_collisions
    })
    
    csv_filename = f"alpha_{alpha}_results.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\n✅ Evaluation complete. CSV saved: {csv_filename}")
    print(f"   Avg Reward: {np.mean(raw_rewards):.2f}")
    print(f"   Avg Captures: {np.mean(raw_total_captures):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_cycles", type=int, default=50) # Added as CLI argument
    args = parser.parse_args()

    evaluate_model(alpha=args.alpha, episodes=args.episodes, max_cycles=args.max_cycles)
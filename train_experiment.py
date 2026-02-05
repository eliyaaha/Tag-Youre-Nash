import argparse
import os
import time
from stable_baselines3 import PPO
from Helpers.RewardSharing import *
from Helpers.Enviroment import *

def run_experiment(alpha, total_timesteps=100000, rounds=10):
    """
    Runs the full adversarial training experiment for a specific alpha.
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT | Alpha: {alpha} ")
    print(f"{'='*60}")
    
    # 1. Create separate environments for Predator and Prey models
    env_pred, _ = create_env(alpha=alpha)
    env_prey, _ = create_env(alpha=alpha)
    
    # 2. Initialize PPO Models
    print(f"Create PPO models on device: {DEVICE}")
    predator_model = PPO("MlpPolicy", env_pred, verbose=0, learning_rate=0.0003, device=DEVICE)
    prey_model = PPO("MlpPolicy", env_prey, verbose=0, learning_rate=0.0003, device=DEVICE)
    
    # 3. Adversarial Training Loop (Self-Play)
    for i in range(rounds):
        start_time = time.time()
        print(f"\n🔄 Round {i+1}/{rounds} in progress...")
        
        # Train Predators
        predator_model.learn(total_timesteps=total_timesteps)
        print(f"   > Predators trained ({total_timesteps} steps)")
        
        # Train Prey (Adapting to new predators)
        prey_model.learn(total_timesteps=total_timesteps)
        print(f"   > Prey trained ({total_timesteps} steps)")
        
        elapsed = time.time() - start_time
        print(f"   ⏱️ Round finished in {elapsed:.2f} seconds.")
    
    # Save Models
    os.makedirs("models", exist_ok=True)
    pred_path = f"./models/predator_alpha_{alpha}"
    prey_path = f"./models/prey_alpha_{alpha}"
    
    predator_model.save(pred_path)
    prey_model.save(prey_path)
    
    print(f"\n✅ Experiment Complete. Models saved to 'models/' folder based on Alpha {alpha}.")

if __name__ == "__main__":
    # Setup argument parser to accept alpha from CLI
    parser = argparse.ArgumentParser(description="Train MARL Agents with Specific Alpha")
    
    # Alpha is required
    parser.add_argument("--alpha", type=float, required=True, help="The cooperation parameter (0.0 to 1.0)")
    
    # Optional arguments (with defaults matching your notebook)
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per round")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")

    args = parser.parse_args()

    # Run the experiment
    run_experiment(alpha=args.alpha, total_timesteps=args.timesteps, rounds=args.rounds)
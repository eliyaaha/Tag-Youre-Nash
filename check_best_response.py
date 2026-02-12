import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from Helpers.Enviroment import create_env
from Helpers.ne_utils import snapshot_world_state, simulate_deviation
from Helpers.Logger import ExperimentLogger  # Import the logger
import check_ne


def infer_n_actions(env):
    """
    Helper to infer the number of actions from the environment's action space.
    """
    try:
        a_space = env.action_space
        import gym
        if isinstance(a_space, gym.spaces.Discrete):
            return a_space.n
        elif isinstance(a_space, gym.spaces.MultiDiscrete):
            return int(a_space.nvec[0])
    except Exception:
        pass
    return 5


def run(alpha, max_cycles=50, sample=100):
    # Initialize the logger
    logger = ExperimentLogger(experiment_name="BestResponseCheck", alpha=alpha)
    
    results_dir = os.path.join("results", f"alpha_{alpha}")
    os.makedirs(results_dir, exist_ok=True)
    stall_csv = os.path.join(results_dir, "stalling_candidates.csv")

    logger.info(f"🚀 Starting Best Response check for Alpha: {alpha}")
    logger.info("Computing stalling_candidates from trajectories...")

    traj_csv = os.path.join(results_dir, "trajectories.csv")
    if not os.path.exists(traj_csv):
        logger.error(f"Trajectories CSV missing: {traj_csv}. Try running evaluation first!")
        raise FileNotFoundError(f"Trajectories CSV missing: {traj_csv}")
    
    try:
        # Find stalling candidates
        df_stall = check_ne.find_stalling(traj_csv, threshold=1e-1, max_per_episode=1)
        df_stall.to_csv(stall_csv, index=False)
        logger.info(f"Found {len(df_stall)} stalling candidates.")
    except Exception as e:
        logger.error(f"Failed to find stalling candidates: {str(e)}")
        return

    # Load the predator model
    model_path = f"./models/predator_alpha_{alpha}"
    if not os.path.exists(model_path):
        logger.error(f"Predator model not found: {model_path}. Try training a model first!")
        raise FileNotFoundError(f"Predator model not found: {model_path}")
    
    pred_model = PPO.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Create environment to roll episodes to the target step
    env, possible_agents, _ = create_env(alpha=alpha, max_cycles=max_cycles, eval=False)

    n_samples = min(len(df_stall), sample)
    df_sample = df_stall.sample(n_samples, random_state=0).reset_index(drop=True)
    logger.info(f"Sampling {n_samples} states for evaluation.")

    br_results = []
    ne_count = 0

    for idx, row in df_sample.iterrows():
        agent = row['agent name']
        ep = int(row['episode id'])
        cycle = int(row['cycle id'])

        # Reset and rollout the environment to the target episode/cycle
        obs = env.reset()
        for t in range(1, cycle + 1):
            action, _ = pred_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

        # Snapshot the world state at this point
        seed_snapshot = snapshot_world_state(env)
        n_actions = infer_n_actions(env)

        # Calculate Baseline action and reward
        baseline_action_arr, _ = pred_model.predict(obs, deterministic=True)
        baseline_a = int(baseline_action_arr[possible_agents.index(agent)])
        baseline_reward = simulate_deviation(alpha, seed_snapshot, agent, baseline_a, pred_model, possible_agents, max_cycles, None)

        # Check for deviations (Best Response calculation)
        best_reward = baseline_reward
        best_action = baseline_a
        
        for a in range(n_actions):
            if a == baseline_a: 
                continue
            try:
                r = simulate_deviation(alpha, seed_snapshot, agent, a, pred_model, possible_agents, max_cycles, None)
            except Exception:
                r = -np.inf
            
            if r > best_reward:
                best_reward = r
                best_action = a

        delta = best_reward - baseline_reward
        is_ne = delta <= 1e-8
        if is_ne: 
            ne_count += 1

        # Log progress for specific sample
        logger.info(f"[{idx+1}/{n_samples}] Ep {ep} Cycle {cycle} | Agent: {agent} | Delta: {delta:.4f} | Is NE: {is_ne}")

        br_results.append({
            'agent name': agent,
            'episode': ep,
            'cycle': cycle,
            'baseline_action': int(baseline_a),
            'baseline_reward': float(baseline_reward),
            'best_action': int(best_action),
            'best_reward': float(best_reward),
            'delta': float(delta),
            'is_ne': bool(is_ne)
        })

        # Save results incrementally
        pd.DataFrame(br_results).to_csv(os.path.join(results_dir, 'best_response_checks.csv'), index=False)

    # Log summary statistics
    ne_percentage = (ne_count / n_samples) * 100 if n_samples > 0 else 0
    logger.info(f"✅ Finished BR checks. NE Stability: {ne_percentage:.2f}% ({ne_count}/{n_samples})")
    logger.info(f"Results saved to {os.path.join(results_dir, 'best_response_checks.csv')}")


def main():
    parser = argparse.ArgumentParser(description='Run best-response single-step checks for stalling states')
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--max_cycles', type=int, default=50)
    parser.add_argument('--sample', type=int, default=50)
    args = parser.parse_args()
    
    run(alpha=args.alpha, max_cycles=args.max_cycles, sample=args.sample)


if __name__ == '__main__':
    main()
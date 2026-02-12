import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from Helpers.Enviroment import create_env
from Helpers.ne_utils import snapshot_world_state, simulate_deviation
from Helpers.Logger import ExperimentLogger
import Evaluation.check_ne as check_ne
from Evaluation.create_ne_table import generate_ne_table

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
        # Find stalling candidates using the logic from check_ne
        df_stall = check_ne.find_stalling(traj_csv, threshold=1e-1, max_per_episode=1)
        df_stall.to_csv(stall_csv, index=False)
        logger.info(f"Found {len(df_stall)} stalling candidates.")
    except Exception as e:
        logger.error(f"Failed to find stalling candidates: {str(e)}")
        return

    # Load the predator model
    model_path = f"./models/predator_alpha_{alpha}"
    if not os.path.exists(model_path):
        logger.error(f"Predator model not found: {model_path}.")
        raise FileNotFoundError(f"Predator model not found: {model_path}")
    
    pred_model = PPO.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Create environment
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

        # Rollout to the target state
        obs = env.reset()
        for t in range(1, cycle + 1):
            action, _ = pred_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

        # Snapshot and check deviations
        seed_snapshot = snapshot_world_state(env)
        n_actions = infer_n_actions(env)

        # Baseline reward calculation
        baseline_action_arr, _ = pred_model.predict(obs, deterministic=True)
        baseline_a = int(baseline_action_arr[possible_agents.index(agent)])
        baseline_reward = simulate_deviation(alpha, seed_snapshot, agent, baseline_a, pred_model, possible_agents, max_cycles, None)

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
        is_ne = delta <= 1e-1
        if is_ne: 
            ne_count += 1

        logger.info(f"[{idx+1}/{n_samples}] Ep {ep} Cycle {cycle} | Delta: {delta:.4f} | Is NE: {is_ne}")

        br_results.append({
            'agent name': agent,
            'episode': ep,
            'cycle': cycle,
            'baseline_reward': float(baseline_reward),
            'best_reward': float(best_reward),
            'delta': float(delta),
            'is_ne': bool(is_ne)
        })

    # --- Analysis Section ---
    df_results = pd.DataFrame(br_results)
    mean_regret = df_results['delta'].mean()
    max_regret = df_results['delta'].max()
    violation_rate = (df_results['is_ne'] == False).mean() * 100

    # Save detailed summary to CSV
    summary_path = os.path.join(results_dir, 'ne_analysis_summary.csv')
    pd.DataFrame([{
        "alpha": alpha,
        "mean_regret": mean_regret,
        "max_regret": max_regret,
        "violation_rate_percent": violation_rate,
        "samples": n_samples
    }]).to_csv(summary_path, index=False)

    # Save full results
    df_results.to_csv(os.path.join(results_dir, 'best_response_checks.csv'), index=False)

    # Final Logging
    logger.info(f"✅ Analysis Complete for Alpha {alpha}")
    logger.info(f"📊 Mean Regret: {mean_regret:.6f} | Violation Rate: {violation_rate:.2f}%")
    logger.info(f"📁 Summary saved: {summary_path}")
    logger.info(f"📁 Full results saved: {os.path.join(results_dir, 'best_response_checks.csv')}")


def main():
    parser = argparse.ArgumentParser(description='Run best-response single-step checks with full NE analysis')
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--max_cycles', type=int, default=50)
    parser.add_argument('--sample', type=int, default=50)
    args = parser.parse_args()
    
    run(alpha=args.alpha, max_cycles=args.max_cycles, sample=args.sample)


if __name__ == '__main__':
    main()
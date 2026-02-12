import argparse
from Helpers.Logger import ExperimentLogger
from Helpers.Enviroment import set_seed 
from evaluate_experiment import evaluate_model
from train_experiment import run_experiment as train_model

# Import the best response logic (Ensure the filename matches check_best_response.py)
from check_best_response import run as run_best_response_check

def main():
    parser = argparse.ArgumentParser(description="MARL Experiment Runner: Batch Processing Alpha Values")
    
    # Core Experiment Arguments
    parser.add_argument("--mode", type=str, nargs='+', choices=["train", "eval", "br", "all"], 
                        default=["all"], 
                        help="Execution modes: train, eval, br. Default is 'all'.") 
    
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0], 
                        help="List of alpha values to process (e.g., --alphas 0.0 0.5 1.0)")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Training Arguments
    parser.add_argument("--timesteps", type=int, default=100000, 
                        help="Total timesteps to train each model")
    
    parser.add_argument("--rounds", type=int, default=10, 
                        help="Total rounds to train each model")
    
    # Evaluation & Best Response Arguments
    parser.add_argument("--episodes", type=int, default=50, 
                        help="Number of episodes for evaluation")
    
    parser.add_argument("--max_cycles", type=int, default=50, 
                        help="Maximum steps per episode")
    
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of states to sample for Best Response check")

    args = parser.parse_args()

    # 1. Global Setup
    set_seed(args.seed)
    
    # Initialize a master logger for the entire batch run
    master_logger = ExperimentLogger(experiment_name=f"Batch_{args.mode}", alpha="ALL")
    master_logger.info(f"🚀 Starting Batch Run | Mode: {args.mode} | Alphas: {args.alphas}")

    if "all" in args.mode:
        active_modes = ["train", "eval", "br"]
    else:
        active_modes = args.mode

    # 2. Iterate through each alpha value
    for alpha in args.alphas:
        master_logger.info(f"\n{'='*40}\nProcessing ALPHA: {alpha}\n{'='*40}")

        # --- Phase 1: Training ---
        if args.mode in active_modes:
            master_logger.info(f"Starting training for alpha {alpha}...")
            try:
                train_model(alpha=alpha, total_timesteps=args.timesteps, rounds=args.rounds)
                master_logger.info(f"Training completed for alpha {alpha}")
            except Exception as e:
                master_logger.error(f"Training FAILED for alpha {alpha}: {str(e)}")
                continue # Skip to next alpha if training fails

        # --- Phase 2: Evaluation ---
        if args.mode in active_modes:
            master_logger.info(f"Starting evaluation for alpha {alpha}...")
            try:
                evaluate_model(
                    alpha=alpha, 
                    episodes=args.episodes, 
                    max_cycles=args.max_cycles
                )
                master_logger.info(f"Evaluation completed for alpha {alpha}")
            except Exception as e:
                master_logger.error(f"Evaluation FAILED for alpha {alpha}: {str(e)}")

        # --- Phase 3: Best Response Check ---
        if args.mode in active_modes:
            master_logger.info(f"Starting Best-Response check for alpha {alpha}...")
            try:
                # Call the run function from your new script
                run_best_response_check(
                    alpha=alpha, 
                    max_cycles=args.max_cycles, 
                    sample=args.sample
                )
                master_logger.info(f"Best-Response check completed for alpha {alpha}")
            except Exception as e:
                master_logger.error(f"Best-Response check FAILED for alpha {alpha}: {str(e)}")

    master_logger.info(f"\n✨ ALL PROCESSES COMPLETED FOR ALPHAS: {args.alphas}")

if __name__ == "__main__":
    main()
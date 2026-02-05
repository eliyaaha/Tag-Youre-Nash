import argparse
from Helpers.Logger import ExperimentLogger
from Helpers.Enviroment import set_seed 
from evaluate_experiment import evaluate_model
from train_experiment import run_experiment as train_model

def main():
    parser = argparse.ArgumentParser(description="MARL Experiment Runner: Batch Processing Alpha Values")
    
    # Core Experiment Arguments
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], required=True,
                        help="Execution mode: train, eval, or both.")
    
    # Changed to accept a list of floats
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0], 
                        help="List of alpha values to process (e.g., --alphas 0.0 0.5 1.0)")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Training Arguments
    parser.add_argument("--timesteps", type=int, default=100000, 
                        help="Total timesteps to train each model")
    
    # Evaluation Arguments
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes for evaluation")
    
    parser.add_argument("--max_cycles", type=int, default=25, 
                        help="Maximum steps per episode")

    args = parser.parse_args()

    # 1. Global Setup
    set_seed(args.seed)
    
    # Initialize a master logger for the entire batch run
    master_logger = ExperimentLogger(experiment_name=f"Batch_{args.mode}", alpha="ALL")
    master_logger.info(f"🚀 Starting Batch Run | Alphas to process: {args.alphas}")

    # 2. Iterate through each alpha value
    for alpha in args.alphas:
        master_logger.info(f"\n{'='*40}\nProcessing ALPHA: {alpha}\n{'='*40}")

        # --- Phase 1: Training ---
        if args.mode in ["train", "both"]:
            master_logger.info(f"Starting training for alpha {alpha}...")
            try:
                train_model(alpha=alpha, timesteps=args.timesteps, seed=args.seed)
                master_logger.info(f"Training completed for alpha {alpha}")
            except Exception as e:
                master_logger.error(f"Training FAILED for alpha {alpha}: {str(e)}")
                continue # Skip to next alpha if training fails

        # --- Phase 2: Evaluation ---
        if args.mode in ["eval", "both"]:
            master_logger.info(f"Starting evaluation for alpha {alpha}...")
            try:
                # Call our evaluation function
                evaluate_model(
                    alpha=alpha, 
                    episodes=args.episodes, 
                    max_cycles=args.max_cycles
                )
                master_logger.info(f"Evaluation completed for alpha {alpha}")
            except Exception as e:
                master_logger.error(f"Evaluation FAILED for alpha {alpha}: {str(e)}")

    master_logger.info(f"\n✨ ALL ALPHAS PROCESSED: {args.alphas}")

if __name__ == "__main__":
    main()
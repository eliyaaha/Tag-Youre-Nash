import os
import argparse
import pandas as pd
import numpy as np

def find_stalling(trajectory_csv, threshold=1e-1, max_per_episode=None):
    df = pd.read_csv(trajectory_csv)
    # Ensure correct column names
    expected_cols = {"agent name", "episode id", "cycle id", "x coordinate", "y coordinate"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Trajectory CSV missing expected columns: {expected_cols - set(df.columns)}")

    results = []

    # Group by agent and episode, compute velocity norms
    gb = df.sort_values(["agent name", "episode id", "cycle id"]).groupby(["agent name", "episode id"])
    for (agent, ep), group in gb:
        coords = group[["x coordinate", "y coordinate"]].to_numpy()
        cycles = group["cycle id"].to_numpy()
        if coords.shape[0] < 2:
            continue
        # velocity at time t corresponds to pos_t - pos_{t-1}, we mark stalling at time t
        diffs = coords[1:] - coords[:-1]
        norms = np.linalg.norm(diffs, axis=1)
        for i, norm in enumerate(norms, start=1):
            if cycles[i] >= 5 and norm <= threshold:
                row = {
                    "agent name": agent,
                    "episode id": int(ep),
                    "cycle id": int(cycles[i]),
                    "x coordinate": float(coords[i][0]),
                    "y coordinate": float(coords[i][1]),
                    "prev_x": float(coords[i-1][0]),
                    "prev_y": float(coords[i-1][1]),
                    "speed": float(norm)
                }
                results.append(row)
                if max_per_episode is not None and sum(1 for r in results if r["episode id"] == ep) >= max_per_episode:
                    break

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Find stalling states from trajectories CSV")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha folder to read trajectories from")
    parser.add_argument("--threshold", type=float, default=1e-1, help="Speed threshold for stalling")
    parser.add_argument("--max_per_episode", type=int, default=1, help="Max stalling candidates per episode to keep")
    args = parser.parse_args()

    results_dir = os.path.join("results", f"alpha_{args.alpha}")
    traj_path = os.path.join(results_dir, "trajectories.csv")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectories CSV not found: {traj_path}")

    out_dir = results_dir
    os.makedirs(out_dir, exist_ok=True)

    df_stall = find_stalling(traj_path, threshold=args.threshold, max_per_episode=args.max_per_episode)
    out_csv = os.path.join(out_dir, "stalling_candidates.csv")
    df_stall.to_csv(out_csv, index=False)
    print(f"Found {len(df_stall)} stalling candidates. Saved to: {out_csv}")


if __name__ == "__main__":
    main()

"""
Run DQN (stable-baselines3) once, on the same frame budget as every other method.

Usage:
    python -m experiments.run_dqn --budget 1000000 --seed 0

--n-seeds is accepted for a uniform command line but unused: RL learns from
every step of every episode instead of scoring candidates on seed sets.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from rl.sb3_agents import run_rl, DEFAULTS

D = DEFAULTS["dqn"]


def main():
    ap = base_parser("DQN")
    ap.add_argument("--lr", type=float, default=D["learning_rate"])
    ap.add_argument("--buffer", type=int, default=D["buffer_size"], help="replay buffer size (steps)")
    ap.add_argument("--learning-starts", type=int, default=D["learning_starts"],
                    help="random steps before training starts")
    ap.add_argument("--target-update", type=int, default=D["target_update_interval"],
                    help="steps between target-network syncs")
    ap.add_argument("--explore-frac", type=float, default=D["exploration_fraction"],
                    help="fraction of training over which epsilon decays 1 -> final")
    ap.add_argument("--final-eps", type=float, default=D["exploration_final_eps"])
    args = ap.parse_args()

    env, budget, log, tag = setup(
        "dqn", args, f"lr{args.lr}_buf{args.buffer // 1000}k_xf{args.explore_frac}")
    params, steps, _ = run_rl("dqn", env, budget, log, seed=args.seed,
                              learning_rate=args.lr, buffer_size=args.buffer,
                              learning_starts=args.learning_starts,
                              target_update_interval=args.target_update,
                              exploration_fraction=args.explore_frac,
                              exploration_final_eps=args.final_eps)
    finish("dqn", env, params, steps, budget, log, tag)


if __name__ == "__main__":
    main()

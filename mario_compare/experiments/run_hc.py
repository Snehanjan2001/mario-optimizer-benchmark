"""
Run the (1+1) hill climber once.

Usage:
    python -m experiments.run_hc --budget 1000000 --seed 0
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from optimizers.hill_climber import run_hill_climber


def main():
    ap = base_parser("(1+1) hill climber")
    ap.add_argument("--sigma", type=float, default=0.05, help="mutation size")
    args = ap.parse_args()

    env, budget, log, tag = setup("hc", args, f"sig{args.sigma}")
    params, it = run_hill_climber(env, budget, log, sigma=args.sigma,
                                  n_seeds=args.n_seeds, seed=args.seed)
    finish("hc", env, params, it, budget, log, tag)


if __name__ == "__main__":
    main()

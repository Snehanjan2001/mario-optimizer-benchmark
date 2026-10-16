"""
Run novelty search (GA + novelty/fitness blend) once.

Usage:
    python -m experiments.run_novelty --budget 1000000 --seed 0
    python -m experiments.run_novelty --novelty-weight 1.0     # pure novelty search
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from optimizers.novelty import run_novelty


def main():
    ap = base_parser("novelty search")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--no-crossover", action="store_true")
    ap.add_argument("--novelty-weight", type=float, default=0.5,
                    help="1 = pure novelty, 0 = plain GA")
    args = ap.parse_args()

    env, budget, log, tag = setup(
        "novelty", args, f"pop{args.pop}_sig{args.sigma}_nw{args.novelty_weight}"
                         f"{'' if args.no_crossover else '_xover'}")
    params, it = run_novelty(env, budget, log, pop_size=args.pop, sigma=args.sigma,
                             crossover=not args.no_crossover,
                             novelty_weight=args.novelty_weight,
                             n_seeds=args.n_seeds, seed=args.seed)
    finish("novelty", env, params, it, budget, log, tag)


if __name__ == "__main__":
    main()

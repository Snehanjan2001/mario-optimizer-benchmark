"""
Run the cross-entropy method once.

Usage:
    python -m experiments.run_cem --budget 1000000 --seed 0
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from optimizers.cem import run_cem


def main():
    ap = base_parser("cross-entropy method")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite-frac", type=float, default=0.2, help="fraction refit each iteration")
    ap.add_argument("--extra-std", type=float, default=0.02, help="noise floor on the std")
    args = ap.parse_args()

    env, budget, log, tag = setup(
        "cem", args, f"pop{args.pop}_elite{args.elite_frac}_xstd{args.extra_std}")
    params, it = run_cem(env, budget, log, pop_size=args.pop, elite_frac=args.elite_frac,
                         extra_std=args.extra_std, n_seeds=args.n_seeds, seed=args.seed)
    finish("cem", env, params, it, budget, log, tag)


if __name__ == "__main__":
    main()

"""
Run CMA-ES once (sep-CMA-ES, diagonal covariance, unless --full).

Usage:
    python -m experiments.run_cmaes --budget 1000000 --seed 0
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from optimizers.cmaes import run_cmaes


def main():
    ap = base_parser("CMA-ES")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma0", type=float, default=0.1, help="initial step size")
    ap.add_argument("--full", action="store_true", help="full covariance (slow at 3990 dims)")
    args = ap.parse_args()

    env, budget, log, tag = setup(
        "cmaes", args, f"pop{args.pop}_sig{args.sigma0}{'_full' if args.full else ''}")
    params, it = run_cmaes(env, budget, log, pop_size=args.pop, sigma0=args.sigma0,
                           diagonal=not args.full, n_seeds=args.n_seeds, seed=args.seed)
    finish("cmaes", env, params, it, budget, log, tag)


if __name__ == "__main__":
    main()

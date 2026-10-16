"""
Run MAP-Elites once.

Usage:
    python -m experiments.run_mapelites --budget 1000000 --seed 0
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from optimizers.map_elites import run_map_elites


def main():
    ap = base_parser("MAP-Elites")
    ap.add_argument("--batch", type=int, default=24, help="children per iteration")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--no-crossover", action="store_true")
    args = ap.parse_args()

    env, budget, log, tag = setup(
        "mapelites", args, f"batch{args.batch}_sig{args.sigma}"
                           f"{'' if args.no_crossover else '_xover'}")
    params, it = run_map_elites(env, budget, log, batch=args.batch, sigma=args.sigma,
                                crossover=not args.no_crossover,
                                n_seeds=args.n_seeds, seed=args.seed)
    finish("mapelites", env, params, it, budget, log, tag)


if __name__ == "__main__":
    main()

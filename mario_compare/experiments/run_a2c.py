"""
Run A2C (stable-baselines3) once, on the same frame budget as every other method.

Usage:
    python -m experiments.run_a2c --budget 1000000 --seed 0

--n-seeds is accepted for a uniform command line but unused: RL learns from
every step of every episode instead of scoring candidates on seed sets.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from rl.sb3_agents import run_rl, DEFAULTS

D = DEFAULTS["a2c"]


def main():
    ap = base_parser("A2C")
    ap.add_argument("--lr", type=float, default=D["learning_rate"])
    ap.add_argument("--n-steps", type=int, default=D["n_steps"], help="steps per update")
    ap.add_argument("--ent-coef", type=float, default=D["ent_coef"], help="entropy bonus")
    args = ap.parse_args()

    env, budget, log, tag = setup("a2c", args, f"lr{args.lr}_nst{args.n_steps}_ent{args.ent_coef}")
    params, steps, _ = run_rl("a2c", env, budget, log, seed=args.seed,
                              learning_rate=args.lr, n_steps=args.n_steps,
                              ent_coef=args.ent_coef)
    finish("a2c", env, params, steps, budget, log, tag)


if __name__ == "__main__":
    main()

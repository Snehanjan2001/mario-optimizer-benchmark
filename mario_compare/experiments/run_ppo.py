"""
Run PPO (stable-baselines3) once, on the same frame budget as every other method.

Usage:
    python -m experiments.run_ppo --budget 1000000 --seed 0
    python -m experiments.run_ppo --lr 1e-4 --ent-coef 0.02

--n-seeds is accepted for a uniform command line but unused: RL learns from
every step of every episode instead of scoring candidates on seed sets.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import base_parser, setup, finish
from rl.sb3_agents import run_rl, DEFAULTS

D = DEFAULTS["ppo"]


def main():
    ap = base_parser("PPO")
    ap.add_argument("--lr", type=float, default=D["learning_rate"])
    ap.add_argument("--n-steps", type=int, default=D["n_steps"], help="steps collected per update")
    ap.add_argument("--batch-size", type=int, default=D["batch_size"])
    ap.add_argument("--n-epochs", type=int, default=D["n_epochs"], help="passes over each rollout")
    ap.add_argument("--clip", type=float, default=D["clip_range"], help="PPO clip range")
    ap.add_argument("--ent-coef", type=float, default=D["ent_coef"], help="entropy bonus")
    args = ap.parse_args()

    env, budget, log, tag = setup(
        "ppo", args, f"lr{args.lr}_nst{args.n_steps}_ent{args.ent_coef}_clip{args.clip}")
    params, steps, _ = run_rl("ppo", env, budget, log, seed=args.seed,
                              learning_rate=args.lr, n_steps=args.n_steps,
                              batch_size=args.batch_size, n_epochs=args.n_epochs,
                              clip_range=args.clip, ent_coef=args.ent_coef)
    finish("ppo", env, params, steps, budget, log, tag)


if __name__ == "__main__":
    main()

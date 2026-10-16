"""
Run OpenAI-style ES once.

Usage:
    python -m experiments.run_es --budget 1000000 --seed 0
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.mario_env import MarioTileEnv, OBS_MODES
from core.rollout import FrameBudget, evaluate, VAL_SEEDS
from optimizers.es import run_es
from policies.mlp import MLPPolicy
from experiments.run_baselines import log_row, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=12, help="antithetic pairs per iteration")
    ap.add_argument("--sigma", type=float, default=0.02, help="perturbation size")
    ap.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    ap.add_argument("--n-seeds", type=int, default=3, help="episodes per fitness evaluation")
    ap.add_argument("--obs", choices=OBS_MODES, default="grid")
    ap.add_argument("--level", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--sticky-prob", type=float, default=0.2)
    args = ap.parse_args()

    env = MarioTileEnv(level=args.level, sticky_prob=args.sticky_prob, obs_mode=args.obs)
    tag = (f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}_sticky{args.sticky_prob}"
           f"_pairs{args.pairs}_sig{args.sigma}_lr{args.lr}_ns{args.n_seeds}"
           f"_b{args.budget // 1000}k"
           f"{'' if args.obs == 'grid' else '_obs' + args.obs}")
    out = RESULTS / f"es_{tag}.csv"
    log = lambda row: log_row(out, dict(row, sticky_prob=env.sticky_prob), {})

    budget = FrameBudget(args.budget)
    theta, final, it = run_es(env, budget, log, n_pairs=args.pairs, sigma=args.sigma,
                              lr=args.lr, n_seeds=args.n_seeds, seed=args.seed)

    # validation score: use THIS to compare settings. es_heldout is for the final report only.
    policy = MLPPolicy(n_inputs=env.observation_space.shape[0])
    policy.set_params(theta)
    val = evaluate(env, policy, VAL_SEEDS)
    print(f"  [es] validation: fitness={val['fitness']:.1f} x={val['max_x']:.1f}")
    log(dict(method="es_val", iteration=it, frames_used=budget.used, **val))
    log(dict(method="es_heldout", iteration=it, frames_used=budget.used, **final))
    np.save(RESULTS / f"es_best_{tag}.npy", theta)
    env.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

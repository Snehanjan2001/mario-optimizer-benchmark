"""
Run the GA once.

Usage:
    python -m experiments.run_ga --budget 200000 --seed 0
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.mario_env import MarioTileEnv, OBS_MODES
from core.rollout import FrameBudget, evaluate, VAL_SEEDS
from optimizers.ga import run_ga
from policies.mlp import MLPPolicy
from experiments.run_baselines import log_row, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--obs", choices=OBS_MODES, default="grid",
                    help="network input: grid | act | prev | both (see envs/mario_env.py)")
    ap.add_argument("--n-seeds", type=int, default=3, help="episodes per fitness evaluation")
    ap.add_argument("--stall-gens", type=int, default=0,
                    help="raise sigma after N gens without progress (0 = off)")
    ap.add_argument("--level", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--sticky-prob", type=float, default=0.2)
    args = ap.parse_args()

    env = MarioTileEnv(level=args.level, sticky_prob=args.sticky_prob, obs_mode=args.obs)
    # every setting in the name, so parallel runs started in the same second never share a file
    tag = (f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}_sticky{args.sticky_prob}"
           f"_pop{args.pop}_sig{args.sigma}_ns{args.n_seeds}_b{args.budget // 1000}k"
           f"{'_xover' if args.crossover else ''}"
           f"{'_st' + str(args.stall_gens) if args.stall_gens else ''}"
           f"{'' if args.obs == 'grid' else '_obs' + args.obs}")
    out = RESULTS / f"ga_{tag}.csv"
    log = lambda row: log_row(out, dict(row, sticky_prob=env.sticky_prob), {})

    budget = FrameBudget(args.budget)
    best, final, gen = run_ga(env, budget, log, pop_size=args.pop,
                              sigma=args.sigma, crossover=args.crossover,
                              n_seeds=args.n_seeds, seed=args.seed,
                              stall_gens=args.stall_gens)

    # validation score: use THIS to compare settings. ga_heldout is for the final report only.
    policy = MLPPolicy(n_inputs=env.observation_space.shape[0])
    policy.set_params(best)
    val = evaluate(env, policy, VAL_SEEDS)
    print(f"  [ga] validation: fitness={val['fitness']:.1f} x={val['max_x']:.1f}")
    log(dict(method="ga_val", iteration=gen, frames_used=budget.used, **val))
    log(dict(method="ga_heldout", iteration=gen, frames_used=budget.used, **final))
    np.save(RESULTS / f"ga_best_{tag}.npy", best)
    env.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

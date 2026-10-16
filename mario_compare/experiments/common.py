"""
Shared plumbing for the per-algorithm runners (run_hc.py, run_cem.py, ...).

Every runner does the same three things, and they live here so no algorithm can
quietly do them differently:

  setup()   -- same env, same budget, one CSV + weights file per run, with every
               setting in the file name
  finish()  -- score the final weights on VAL_SEEDS (for choosing settings) and
               TEST_SEEDS (for the final report), log both, save the weights

(run_ga.py / run_es.py predate this file and do the same steps inline.)
"""

from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np

from envs.mario_env import MarioTileEnv, OBS_MODES
from policies.mlp import MLPPolicy
from core.rollout import FrameBudget, evaluate, VAL_SEEDS, TEST_SEEDS
from experiments.run_baselines import log_row, RESULTS


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--budget", type=int, default=1_000_000, help="emulator frames")
    ap.add_argument("--seed", type=int, default=0, help="run seed")
    ap.add_argument("--n-seeds", type=int, default=3, help="episodes per fitness evaluation")
    ap.add_argument("--obs", choices=OBS_MODES, default="grid")
    ap.add_argument("--level", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--sticky-prob", type=float, default=0.2)
    return ap


def setup(method: str, args, settings: str):
    """Returns (env, budget, log, tag). `settings` is this algorithm's part of the name."""
    env = MarioTileEnv(level=args.level, sticky_prob=args.sticky_prob, obs_mode=args.obs)
    tag = (f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}_sticky{args.sticky_prob}"
           f"_{settings}_ns{args.n_seeds}_b{args.budget // 1000}k"
           f"{'' if args.obs == 'grid' else '_obs' + args.obs}")
    out = RESULTS / f"{method}_{tag}.csv"

    def log(row: dict) -> None:
        log_row(out, dict(row, sticky_prob=env.sticky_prob), {})

    log.out = out
    return env, FrameBudget(args.budget), log, tag


def finish(method: str, env, params: np.ndarray, iterations: int, budget, log, tag: str):
    """Validation + held-out scores for the final weights, then save them."""
    policy = MLPPolicy(n_inputs=env.observation_space.shape[0])
    policy.set_params(params)
    val = evaluate(env, policy, VAL_SEEDS)
    test = evaluate(env, policy, TEST_SEEDS)
    print(f"  [{method}] {iterations} iterations, {budget.used:,} frames")
    print(f"  [{method}] validation: fitness={val['fitness']:.1f} x={val['max_x']:.1f}")
    print(f"  [{method}] held-out:   fitness={test['fitness']:.1f} x={test['max_x']:.1f}")
    log(dict(method=f"{method}_val", iteration=iterations, frames_used=budget.used, **val))
    log(dict(method=f"{method}_heldout", iteration=iterations, frames_used=budget.used, **test))
    np.save(RESULTS / f"{method}_best_{tag}.npy", params)
    env.close()
    print(f"wrote {log.out}")

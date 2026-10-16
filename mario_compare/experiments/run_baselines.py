"""
Step 3 experiment: establish the floors.

Run this before any optimizer. It produces the two reference numbers that make
the rest of the study interpretable:

  1. HOLD-RIGHT   -- what a 3-line script achieves with no learning at all.
                     On 1-1 this is embarrassingly good. Any evolved policy
                     that does not clearly beat it has learned nothing.

  2. RANDOM SEARCH -- sample MLP weight vectors, keep the best. This is the
                     GA with selection, crossover and mutation removed. The
                     gap between GA and this is the only honest measure of
                     what the genetic machinery actually contributes.

Usage:
    python -m experiments.run_baselines --budget 200000
"""

from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.mario_env import MarioTileEnv, OBS_MODES
from policies.mlp import MLPPolicy, HoldRightPolicy, RandomPolicy
from core.rollout import FrameBudget, BudgetExhausted, evaluate, seed_set, TEST_SEEDS, VAL_SEEDS

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


# Fixed column order. Scripted and search rows carry different keys, so letting
# DictWriter take its columns from whichever row comes first misaligns the file.
FIELDS = ["method", "iteration", "frames_used", "sticky_prob",
          "fitness", "fitness_std", "fitness_min", "fitness_max",
          "max_x", "best_x", "flag_rate", "env_reward", "frames", "wall_s"]


def log_row(path: Path, row: dict, header_written: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        # extra keys (e.g. evaluate()'s final_x/final_y) are not columns: ignore them
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        if new and not header_written.get(path):
            w.writeheader()
            header_written[path] = True
        w.writerow(row)


def run_scripted(env, name, policy, seeds, out):
    t0 = time.time()
    res = evaluate(env, policy, seeds)
    res.update(method=name, iteration=0, frames_used=res["frames"],
               wall_s=round(time.time() - t0, 1), sticky_prob=env.sticky_prob)
    log_row(out, res, {})
    print(f"{name:22s} fitness={res['fitness']:8.1f} "
          f"x={res['max_x']:7.1f} best_x={res['best_x']:5d} "
          f"flag={res['flag_rate']:.2f}  ({res['frames']} frames)")
    return res


def run_random_search(env, budget, out, best_path, n_seeds=3, seed=0):
    """Control condition for the GA."""
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    best_fit, best_params, it = -np.inf, policy.get_params(), 0

    try:
        while True:
            policy.set_params(0.1 * rng.standard_normal(policy.n_params))
            res = evaluate(env, policy, seed_set(rng, n_seeds), budget)
            it += 1
            if res["fitness"] > best_fit:
                best_fit, best_params = res["fitness"], policy.get_params()
                print(f"  [random-search] it={it:4d} new best={best_fit:8.1f} "
                      f"x={res['max_x']:7.1f}  budget={budget.fraction:.1%}")
            log_row(out, dict(method="random_search", iteration=it,
                              frames_used=budget.used, **res,
                              sticky_prob=env.sticky_prob), {})
    except BudgetExhausted:
        pass

    np.save(best_path, best_params)
    print(f"  [random-search] {it} candidates evaluated, best={best_fit:.1f}")

    policy.set_params(best_params)
    val = evaluate(env, policy, VAL_SEEDS)
    print(f"  [random-search] validation: fitness={val['fitness']:.1f} x={val['max_x']:.1f}")
    log_row(out, dict(method="random_search_val", iteration=it,
                      frames_used=budget.used, **val,
                      sticky_prob=env.sticky_prob), {})
    final = evaluate(env, policy, TEST_SEEDS)
    print(f"  [random-search] held-out: fitness={final['fitness']:.1f} "
          f"x={final['max_x']:.1f}")
    log_row(out, dict(method="random_search_heldout", iteration=it,
                      frames_used=budget.used, **final,
                      sticky_prob=env.sticky_prob), {})
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200_000,
                    help="emulator frames for random search")
    ap.add_argument("--level", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--sticky-prob", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="random-search RNG seed")
    ap.add_argument("--obs", choices=OBS_MODES, default="grid")
    args = ap.parse_args()

    env = MarioTileEnv(level=args.level, frame_skip=args.frame_skip,
                       sticky_prob=args.sticky_prob, obs_mode=args.obs)
    # one file per run, so rows from different settings never mix
    tag = (f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}_sticky{args.sticky_prob}"
           f"{'' if args.obs == 'grid' else '_obs' + args.obs}")
    out = RESULTS / f"baselines_{tag}.csv"

    print(f"\n=== scripted baselines ({len(TEST_SEEDS)} held-out seeds) ===")
    run_scripted(env, "random_actions", RandomPolicy(0), TEST_SEEDS, out)
    run_scripted(env, "hold_right", HoldRightPolicy(jump_period=0), TEST_SEEDS, out)
    for period in (12, 20, 30, 45):
        run_scripted(env, f"hold_right_jump{period}",
                     HoldRightPolicy(jump_period=period), TEST_SEEDS, out)

    print(f"\n=== random weight search (budget {args.budget:,} frames) ===")
    run_random_search(env, FrameBudget(args.budget), out,
                      RESULTS / f"random_search_best_{tag}.npy", seed=args.seed)

    env.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

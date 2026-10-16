"""
Score every saved model on the validation seeds and compare settings.

Choose settings by the VAL column. The TEST column (the held-out score logged
at the end of each run) is shown only so you can see how much picking by test
flattered a setting -- do not choose by it.

Usage:
    python -m experiments.score_val              # all finished runs
    python -m experiments.score_val --workers 8
"""

from __future__ import annotations
import argparse, csv, re, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.mario_env import MarioTileEnv, obs_size
from policies.mlp import MLPPolicy
from core.rollout import evaluate, VAL_SEEDS

RESULTS = Path(__file__).resolve().parents[1] / "results"

# methods whose runners use experiments/common.py: file names are
#   <method>_best_<time>_seed<N>_sticky<p>_<settings...>_ns<N>_b<N>k[_obs<mode>].npy
GENERIC = {"hc": "hill climber", "cem": "CEM", "cmaes": "CMA-ES", "novelty": "novelty",
           "mapelites": "MAP-Elites", "ppo": "PPO", "a2c": "A2C", "dqn": "DQN"}


def obs_mode_of(path: Path) -> str:
    m = re.search(r"_obs([a-z]+)", path.stem)
    return m.group(1) if m else "grid"


def heldout_row(csv_path: Path, method: str):
    if not csv_path.exists():
        return None
    rows = [r for r in csv.DictReader(csv_path.open()) if r["method"] == method]
    return rows[-1] if rows else None


def find_models():
    """[(setting label, run seed, npy path, test fitness)] for every finished run."""
    models = []
    for npy in RESULTS.glob("ga_best_*.npy"):
        row = heldout_row(RESULTS / npy.name.replace("ga_best_", "ga_", 1)
                          .replace(".npy", ".csv"), "ga_heldout")
        if row is None:
            continue
        name = npy.stem
        seed = int(re.search(r"_seed(\d+)", name).group(1))
        pop = re.search(r"_pop(\d+)", name)
        sig = re.search(r"_sig([\d.]+)", name)
        bud = re.search(r"_b(\d+)k", name)
        ns = re.search(r"_ns(\d+)", name)
        # runs from before settings were put in file names used the defaults
        label = " ".join(filter(None, [
            f"GA pop{pop.group(1) if pop else 24}",
            f"sig{sig.group(1) if sig else 0.03}",
            f"ns{ns.group(1)}" if ns and ns.group(1) != "3" else "",
            (f"{int(bud.group(1)) // 1000}M" if int(bud.group(1)) >= 1000 else f"{bud.group(1)}k")
            if bud else "1M",
            "+xover" if "_xover" in name else "",
            f"stall{re.search(r'_st(\d+)', name).group(1)}" if re.search(r"_st(\d+)", name) else "",
            "" if obs_mode_of(npy) == "grid" else f"obs={obs_mode_of(npy)}",
        ]))
        models.append((label, seed, npy, float(row["fitness"])))

    for npy in RESULTS.glob("es_best_*.npy"):
        row = heldout_row(RESULTS / npy.name.replace("es_best_", "es_", 1)
                          .replace(".npy", ".csv"), "es_heldout")
        if row is None:
            continue
        name = npy.stem
        get = lambda key: re.search(rf"_{key}([\d.]+?)(?:_|$)", name).group(1)
        bud = int(re.search(r"_b(\d+)k", name).group(1))
        label = " ".join(filter(None, [
            f"ES pairs{get('pairs')} sig{get('sig')} lr{get('lr')}",
            f"ns{get('ns')}" if get("ns") != "3" else "",
            f"{bud // 1000}M" if bud >= 1000 else f"{bud}k",
            "" if obs_mode_of(npy) == "grid" else f"obs={obs_mode_of(npy)}",
        ]))
        models.append((label, int(get("seed")), npy, float(row["fitness"])))

    for method, display in GENERIC.items():
        for npy in RESULTS.glob(f"{method}_best_*.npy"):
            row = heldout_row(RESULTS / npy.name.replace(f"{method}_best_", f"{method}_", 1)
                              .replace(".npy", ".csv"), f"{method}_heldout")
            if row is None:
                continue
            parts = npy.stem.split("_best_", 1)[1].split("_")[1:]    # drop the timestamp
            seed = int(parts[0][4:])
            settings = []
            for p in parts[2:]:                                    # after seed, sticky
                if p == "ns3":
                    continue
                m = re.fullmatch(r"b(\d+)k", p)
                if m:
                    k = int(m.group(1))
                    p = f"{k // 1000}M" if k >= 1000 else f"{k}k"
                elif p.startswith("obs"):
                    p = f"obs={p[3:]}"
                settings.append(p)
            models.append((f"{display} " + " ".join(settings), seed, npy, float(row["fitness"])))

    for npy in RESULTS.glob("random_search_best_*.npy"):
        row = heldout_row(RESULTS / npy.name.replace("random_search_best_", "baselines_", 1)
                          .replace(".npy", ".csv"), "random_search_heldout")
        if row is None:
            continue
        m = re.search(r"_seed(\d+)", npy.stem)
        frames = int(row["frames_used"])
        label = f"random search {round(frames / 1e6)}M" if frames >= 500_000 \
            else f"random search {frames // 1000}k"
        if obs_mode_of(npy) != "grid":
            label += f" obs={obs_mode_of(npy)}"
        models.append((label, int(m.group(1)) if m else 0, npy, float(row["fitness"])))
    return models


def score(npy: Path) -> float:
    mode = obs_mode_of(npy)
    env = MarioTileEnv(obs_mode=mode)
    policy = MLPPolicy(n_inputs=obs_size(mode))
    policy.set_params(np.load(npy))
    res = evaluate(env, policy, VAL_SEEDS)
    env.close()
    return res["fitness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    models = find_models()
    print(f"scoring {len(models)} models on {len(VAL_SEEDS)} validation seeds ...")
    with Pool(args.workers) as pool:
        vals = pool.map(score, [m[2] for m in models])

    by_setting = defaultdict(list)
    for (label, seed, npy, test), val in zip(models, vals):
        by_setting[label].append((seed, val, test))

    print(f"\n{'setting':40s} {'runs':>4s} {'VAL mean ± std':>16s}   {'(test mean)':>11s}   per-run val")
    for label, runs in sorted(by_setting.items(), key=lambda kv: -np.mean([r[1] for r in kv[1]])):
        runs.sort()
        v = np.array([r[1] for r in runs])
        t = np.array([r[2] for r in runs])
        per_run = "  ".join(f"s{s}:{val:.0f}" for s, val, _ in runs)
        print(f"{label:40s} {len(runs):4d} {v.mean():8.1f} ± {v.std():5.1f}   ({t.mean():9.1f})   {per_run}")


if __name__ == "__main__":
    main()

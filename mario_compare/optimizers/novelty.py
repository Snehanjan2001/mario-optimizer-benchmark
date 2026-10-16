"""
Novelty search, combined with fitness, on top of the GA.

The GA only rewards "got further right". When the whole population is stuck at
the same pipe, they all score the same and selection has nothing to work with.
Novelty search also rewards behaving DIFFERENTLY from what has been seen before
(Lehman & Stanley 2011): ending on top of a pipe, or in a new spot, counts for
something even if it is not yet further right.

Behaviour  = where the episode ended (final x, final y), in tiles, averaged
             over the evaluation seeds.
Novelty    = mean distance to the k nearest behaviours among the current
             population and an archive of past behaviours.
Selection  = tournament on a blend of novelty rank and fitness rank
             (novelty_weight=1 is pure novelty search, 0 is the plain GA).
Elites     = still kept by FITNESS, and the returned network is the fittest,
             so the benchmark measures the same thing as for every method.
"""

from __future__ import annotations
import numpy as np

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted, evaluate, seed_set
from optimizers.ga import tournament, uniform_crossover
from optimizers.es import centered_ranks

TILE = 16.0


def behaviour(res: dict) -> np.ndarray:
    return np.array([res["final_x"] / TILE, res["final_y"] / TILE])


def novelty_scores(bcs: np.ndarray, archive: list, k: int) -> np.ndarray:
    """Mean distance from each behaviour to its k nearest neighbours
    (other population members + archive; never itself)."""
    ref = np.vstack([bcs] + ([np.array(archive)] if archive else []))
    d = np.linalg.norm(bcs[:, None, :] - ref[None, :, :], axis=2)
    d[np.arange(len(bcs)), np.arange(len(bcs))] = np.inf      # exclude self
    k = min(k, ref.shape[0] - 1)
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


def run_novelty(env, budget, log, pop_size=24, n_elite=2, k=3, sigma=0.05,
                crossover=True, novelty_weight=0.5, k_nn=10, archive_prob=0.1,
                n_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    n = policy.n_params

    pop = [0.1 * rng.standard_normal(n) for _ in range(pop_size)]
    archive: list[np.ndarray] = []
    best, gen = pop[0], 0

    try:
        while True:
            seeds = seed_set(rng, n_seeds)
            fits = np.empty(pop_size)
            bcs = np.empty((pop_size, 2))
            for i, genome in enumerate(pop):
                policy.set_params(genome)
                res = evaluate(env, policy, seeds, budget)
                fits[i], bcs[i] = res["fitness"], behaviour(res)

            nov = novelty_scores(bcs, archive, k_nn)
            score = novelty_weight * centered_ranks(nov) + (1 - novelty_weight) * centered_ranks(fits)

            # archive: a random sample of behaviours seen so far (Lehman & Stanley)
            for b in bcs[rng.random(pop_size) < archive_prob]:
                archive.append(b)

            order = np.argsort(fits)[::-1]
            best = pop[order[0]]
            gen += 1
            log(dict(method="novelty", iteration=gen, frames_used=budget.used,
                     fitness=fits.mean(), fitness_std=fits.std(),
                     fitness_min=fits.min(), fitness_max=fits.max()))
            print(f"  [novelty] gen={gen:3d} best={fits.max():7.1f} mean={fits.mean():7.1f} "
                  f"novelty={nov.mean():5.2f} archive={len(archive):3d}  budget={budget.fraction:.1%}")

            children = [pop[i].copy() for i in order[:n_elite]]     # elites by fitness
            while len(children) < pop_size:
                child = pop[tournament(rng, score, k)].copy()       # parents by blended score
                if crossover:
                    child = uniform_crossover(rng, child, pop[tournament(rng, score, k)])
                child += sigma * rng.standard_normal(n)
                children.append(child)
            pop = children
    except BudgetExhausted:
        pass    # the generation that ran out is discarded; best is from the last full one

    return best, gen

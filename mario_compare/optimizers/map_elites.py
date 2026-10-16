"""
MAP-Elites (Mouret & Clune 2015) over flat MLP weights.

Instead of one population chasing one score, keep a GRID of niches, each
holding the best network found for one kind of behaviour. Here a niche is
"where did the episode end": a column of the level (final x, 64-pixel bins)
times a height band (final y, 32-pixel bins). "Died in the first pit",
"stuck on top of the tall pipe" and "reached the second Goomba group" are
different cells, and each keeps its own champion.

Each iteration: pick random champions from the grid, mutate them, score the
batch on shared seeds, and put each child in its cell if the cell is empty or
the child beats the current champion.

Known weakness with noisy fitness: a champion's score comes from one
evaluation and is never re-checked, so lucky networks can hold a cell. That is
exactly what the validation seeds will show.
"""

from __future__ import annotations
import numpy as np

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted, evaluate, seed_set
from optimizers.ga import uniform_crossover

X_BIN, Y_BIN = 64, 32


def cell_of(res: dict) -> tuple[int, int]:
    return int(res["final_x"] // X_BIN), int(res["final_y"] // Y_BIN)


def run_map_elites(env, budget, log, batch=24, sigma=0.05, crossover=True,
                   n_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    n = policy.n_params

    grid: dict[tuple[int, int], tuple[float, np.ndarray]] = {}   # cell -> (fitness, genome)
    it = 0

    try:
        while True:
            seeds = seed_set(rng, n_seeds)
            if not grid:
                # first batch: random networks, same init as every other method
                genomes = [0.1 * rng.standard_normal(n) for _ in range(batch)]
            else:
                cells = list(grid)
                pick = lambda: grid[cells[rng.integers(len(cells))]][1]
                genomes = []
                for _ in range(batch):
                    child = pick().copy()
                    if crossover:
                        child = uniform_crossover(rng, child, pick())
                    genomes.append(child + sigma * rng.standard_normal(n))

            fits = np.empty(batch)
            placed = 0
            for i, g in enumerate(genomes):
                policy.set_params(g)
                res = evaluate(env, policy, seeds, budget)
                fits[i] = res["fitness"]
                c = cell_of(res)
                if c not in grid or fits[i] > grid[c][0]:
                    grid[c] = (fits[i], g)
                    placed += 1

            it += 1
            champion = max(v[0] for v in grid.values())
            log(dict(method="mapelites", iteration=it, frames_used=budget.used,
                     fitness=fits.mean(), fitness_std=fits.std(),
                     fitness_min=fits.min(), fitness_max=champion))
            print(f"  [mapelites] it={it:3d} batch best={fits.max():7.1f} grid best={champion:7.1f} "
                  f"cells={len(grid):3d} placed={placed:2d}  budget={budget.fraction:.1%}")
    except BudgetExhausted:
        pass    # only the child whose evaluation ran out is lost; earlier ones were paid for and placed

    return max(grid.values(), key=lambda v: v[0])[1], it

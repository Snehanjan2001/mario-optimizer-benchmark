"""
Cross-Entropy Method (CEM) over flat MLP weights.

Keeps a Gaussian over weight vectors (a mean and a per-weight std). Each
iteration:

    1. sample a population from the Gaussian
    2. score everyone on the same seeds
    3. refit the Gaussian to the top `elite_frac` of them

The distribution slides toward good regions and narrows as it gains
confidence. Extra noise is added to the std (Szita & Lorincz 2006) so it cannot
collapse to zero before it has found anything.
"""

from __future__ import annotations
import numpy as np

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted, evaluate, seed_set


def run_cem(env, budget, log, pop_size=24, elite_frac=0.2, init_std=0.1,
            extra_std=0.02, n_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    n = policy.n_params
    n_elite = max(2, int(round(pop_size * elite_frac)))

    # mean 0, std 0.1: generation 0 has the same distribution as every other method's init
    mean = np.zeros(n)
    std = np.full(n, init_std)
    it = 0

    try:
        while True:
            seeds = seed_set(rng, n_seeds)
            samples = mean + std * rng.standard_normal((pop_size, n))
            fits = np.empty(pop_size)
            for i, x in enumerate(samples):
                policy.set_params(x)
                fits[i] = evaluate(env, policy, seeds, budget)["fitness"]

            elites = samples[np.argsort(fits)[::-1][:n_elite]]
            mean = elites.mean(axis=0)
            std = np.sqrt(elites.var(axis=0) + extra_std ** 2)

            it += 1
            log(dict(method="cem", iteration=it, frames_used=budget.used,
                     fitness=fits.mean(), fitness_std=fits.std(),
                     fitness_min=fits.min(), fitness_max=fits.max()))
            print(f"  [cem] it={it:3d} best={fits.max():7.1f} mean={fits.mean():7.1f} "
                  f"avg std={std.mean():.4f}  budget={budget.fraction:.1%}")
    except BudgetExhausted:
        pass    # the iteration that ran out is discarded

    # the answer is the distribution's mean, not any single sample
    return mean, it

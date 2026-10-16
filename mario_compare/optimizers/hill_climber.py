"""
(1+1) hill climber over flat MLP weights.

The simplest search that is smarter than random search: keep ONE network,
propose one mutated copy, keep whichever scores better. No population, no
crossover, no gradient estimate.

Its place in the benchmark: GA minus the population. If the GA barely beats
this, the population is not what makes the GA work.
"""

from __future__ import annotations
import numpy as np

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted, evaluate, seed_set


def run_hill_climber(env, budget, log, sigma=0.05, n_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    n = policy.n_params

    parent = 0.1 * rng.standard_normal(n)       # same init as every other method
    it, accepted = 0, 0

    try:
        while True:
            # parent and child are scored on the SAME fresh seeds every iteration.
            # Re-scoring the parent costs frames, but otherwise a parent that got
            # lucky once would block every child forever.
            seeds = seed_set(rng, n_seeds)
            child = parent + sigma * rng.standard_normal(n)

            policy.set_params(parent)
            f_parent = evaluate(env, policy, seeds, budget)["fitness"]
            policy.set_params(child)
            f_child = evaluate(env, policy, seeds, budget)["fitness"]

            # >= (not >): accepting equal children lets the climber drift across
            # flat regions, e.g. while every variant is stuck at the same pipe
            if f_child >= f_parent:
                parent, accepted = child, accepted + 1

            it += 1
            log(dict(method="hc", iteration=it, frames_used=budget.used,
                     fitness=max(f_parent, f_child), fitness_std=abs(f_child - f_parent) / 2,
                     fitness_min=min(f_parent, f_child), fitness_max=max(f_parent, f_child)))
            if it % 20 == 0:
                print(f"  [hc] it={it:4d} parent={f_parent:7.1f} child={f_child:7.1f} "
                      f"accepted={accepted / it:.0%}  budget={budget.fraction:.1%}")
    except BudgetExhausted:
        pass

    return parent, it

"""
CMA-ES over flat MLP weights, via the `cma` package (Hansen's reference implementation).

Like CEM it samples from a Gaussian and moves it toward the better samples,
but it also:
  * learns the SHAPE of the Gaussian (which directions are worth searching),
  * adapts the overall step size sigma from how consistently the mean moves,
  * weights every ranked sample instead of a hard top-20% cut.

Full CMA-ES stores a 3990 x 3990 covariance matrix and decomposes it every few
iterations -- too slow and too data-hungry for ~45 iterations. The default here
is sep-CMA-ES (Ros & Hansen 2008): a diagonal covariance, i.e. one learned
step size per weight. --full switches to the full matrix.
"""

from __future__ import annotations
import numpy as np
import cma

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted, evaluate, seed_set


def run_cmaes(env, budget, log, pop_size=24, sigma0=0.1, diagonal=True, n_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    n = policy.n_params

    es = cma.CMAEvolutionStrategy(np.zeros(n), sigma0, {
        "popsize": pop_size,
        "seed": seed + 1,            # cma treats seed 0 as "random"; +1 keeps runs reproducible
        "CMA_diagonal": diagonal,
        "verbose": -9,
        # never stop early: the frame budget is the only stopping rule, as for every method
        "tolfun": 0, "tolx": 0, "tolfunhist": 0, "tolflatfitness": np.inf,
        "tolstagnation": np.inf, "maxiter": np.inf,
    })
    it = 0

    try:
        while True:
            seeds = seed_set(rng, n_seeds)
            xs = es.ask()
            fits = np.empty(pop_size)
            for i, x in enumerate(xs):
                policy.set_params(x)
                fits[i] = evaluate(env, policy, seeds, budget)["fitness"]
            es.tell(xs, (-fits).tolist())     # cma minimises, we maximise

            it += 1
            log(dict(method="cmaes", iteration=it, frames_used=budget.used,
                     fitness=fits.mean(), fitness_std=fits.std(),
                     fitness_min=fits.min(), fitness_max=fits.max()))
            print(f"  [cmaes] it={it:3d} best={fits.max():7.1f} mean={fits.mean():7.1f} "
                  f"sigma={es.sigma:.4f}  budget={budget.fraction:.1%}")
    except BudgetExhausted:
        pass    # the iteration that ran out is never told to cma

    # the distribution mean is CMA-ES's best estimate of the optimum
    return np.array(es.mean), it

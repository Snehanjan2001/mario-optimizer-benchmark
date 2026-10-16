"""
OpenAI-style Evolution Strategies (Salimans et al. 2017) over flat MLP weights.

The GA keeps a population of competing individuals. ES keeps ONE weight vector
theta and estimates which direction improves it:

    1. perturb theta with Gaussian noise, in +/- pairs (antithetic sampling)
    2. score every perturbation on the same seeds
    3. move theta toward the noise directions that scored well

Same evaluate(), same frame budget, same seeds protocol as the GA -- only the
way new weight vectors are proposed differs.
"""

from __future__ import annotations
import numpy as np

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted, evaluate, seed_set, TEST_SEEDS


def centered_ranks(x: np.ndarray, rng=None) -> np.ndarray:
    """Replace scores by their rank, scaled to [-0.5, 0.5].

    Only the ordering matters, so one lucky 900-point episode cannot drag
    the whole update toward it.

    Ties (common: every candidate stuck at the same pipe scores the same):
      rng given -> broken in random order. ES uses this: on a plateau the
                   update becomes a random step, so theta random-walks until
                   something scores differently. (The reference implementation
                   gets the same effect from argsort's arbitrary order.) With
                   tied ranks averaged instead, a run whose every candidate
                   scored the same never moved again.
      rng None  -> tied scores share their average rank (used by novelty
                   search, where ties should carry no preference).
    """
    if rng is not None:
        order = np.lexsort((rng.random(len(x)), x))     # sort by x, ties by a random key
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
    else:
        ranks = np.empty(len(x))
        ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
        _, groups = np.unique(x, return_inverse=True)
        ranks = (np.bincount(groups, ranks) / np.bincount(groups))[groups]
    return ranks / (len(x) - 1) - 0.5


class Adam:
    """Standard Adam, used for gradient ASCENT (the step is added to theta)."""

    def __init__(self, n: int, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def run_es(env, budget, log, n_pairs=12, sigma=0.02, lr=0.01, weight_decay=0.005,
           n_seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])
    n = policy.n_params

    # same init distribution as GA / random search
    theta = 0.1 * rng.standard_normal(n)
    opt = Adam(n, lr)
    it = 0

    try:
        while True:
            # one seed set per iteration, shared by every perturbation (fair ranking)
            seeds = seed_set(rng, n_seeds)
            noise = rng.standard_normal((n_pairs, n))
            fits = np.empty(2 * n_pairs)
            for i in range(n_pairs):
                for j, sign in enumerate((1.0, -1.0)):
                    policy.set_params(theta + sign * sigma * noise[i])
                    fits[2 * i + j] = evaluate(env, policy, seeds, budget)["fitness"]

            # antithetic estimate: for each pair, (+ rank) - (- rank) says which way is uphill
            w = centered_ranks(fits, rng)
            w = w[0::2] - w[1::2]
            grad = w @ noise / (2 * n_pairs * sigma)
            # weight decay applied outside Adam (AdamW-style): inside, Adam would rescale
            # it to a full lr-sized step whenever the rank gradient is zero (all ties)
            theta = theta + opt.step(grad) - lr * weight_decay * theta

            it += 1
            log(dict(method="es", iteration=it, frames_used=budget.used,
                     fitness=fits.mean(), fitness_std=fits.std(),
                     fitness_min=fits.min(), fitness_max=fits.max()))
            print(f"  [es] it={it:3d} best={fits.max():7.1f} "
                  f"mean={fits.mean():7.1f}  budget={budget.fraction:.1%}")
    except BudgetExhausted:
        pass    # the iteration that ran out is discarded; theta is from the last full update

    policy.set_params(theta)
    final = evaluate(env, policy, TEST_SEEDS)      # held-out, not charged
    print(f"  [es] {it} iterations, held-out: fitness={final['fitness']:.1f} "
          f"x={final['max_x']:.1f}")
    return theta, final, it

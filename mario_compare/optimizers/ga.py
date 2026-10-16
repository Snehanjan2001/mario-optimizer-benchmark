"""
Genetic algorithm over flat MLP weight vectors.

This is run_random_search plus three things: selection, mutation, elitism
(crossover optional). Anything the GA beats random search by is what those
operators contribute.
"""

from __future__ import annotations
import numpy as np

from policies.mlp import MLPPolicy
from core.rollout import BudgetExhausted,evaluate,seed_set,TEST_SEEDS



## Selection and crossover
def tournament(rng,fits:np.ndarray,k:int):
    # pick k random individuals , return the index of the fittest

    idx = rng.choice(len(fits),size = k, replace = False)
    return int(idx[np.argmax(fits[idx])])


def uniform_crossover(rng, a: np.ndarray,b : np.ndarray):
    #Each gene comes from a or b with probability 1/2.
    return np.where(rng.random(a.size)<0.5,a,b)


## Run the main loop

def run_ga(env, budget, log, pop_size = 24, n_elite=2, k = 3 , sigma = 0.03,n_seeds = 3, crossover = False, seed = 0, stall_gens = 0):
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(rng=rng, n_inputs=env.observation_space.shape[0])

    n = policy.n_params

    # same init distribution as random search, so generation 0 is identical to it
    pop = [0.1* rng.standard_normal(n) for _ in range(pop_size)]
    best, gen = pop[0],0
    # adaptive mutation: sig grows while the best score is stuck (stall_gens=0 -> off)
    best_ever, stall, sig = -np.inf, 0, sigma


    try:
        while True:

            seeds = seed_set(rng,n_seeds)
            fits = np.empty(pop_size)
            for i, genome in enumerate(pop):
                policy.set_params(genome)
                fits[i] = evaluate(env,policy,seeds,budget)["fitness"]
            
            order = np.argsort(fits)[::-1]

            best = pop[order[0]]

            gen+=1
            log(dict(method="ga", iteration=gen, frames_used=budget.used,
                     fitness=fits.mean(), fitness_std=fits.std(),
                     fitness_min=fits.min(), fitness_max=fits.max()))
            print(f"  [ga] gen={gen:3d} best={fits.max():7.1f} "
                  f"mean={fits.mean():7.1f}  budget={budget.fraction:.1%}")

            if fits.max() > best_ever + 1:
                best_ever, stall, sig = fits.max(), 0, sigma
            elif stall_gens:
                stall += 1
                if stall >= stall_gens:
                    sig = min(sig * 1.5, 4 * sigma)

            children = [pop[i].copy() for i in order[:n_elite]]
            while len(children) < pop_size:
                child = pop[tournament(rng, fits, k)].copy()
                if crossover:
                    other = pop[tournament(rng, fits, k)]
                    child = uniform_crossover(rng, child, other)
                child += sig * rng.standard_normal(n)
                children.append(child)
            pop = children
    except BudgetExhausted:
        pass    # the generation that ran out is discarded; best is from the last full one

    policy.set_params(best)
    final = evaluate(env, policy, TEST_SEEDS)      # held-out, not charged
    print(f"  [ga] {gen} generations, held-out: fitness={final['fitness']:.1f} "
          f"x={final['max_x']:.1f}")
    return best, final, gen













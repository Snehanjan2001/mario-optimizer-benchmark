# 03 · Genetic algorithm (GA)

**Code:** [`optimizers/ga.py`](../../optimizers/ga.py) → `run_ga`
**Run:** `python -m experiments.run_ga --budget 1000000 --seed 0 --sigma 0.05 --crossover`

## The idea

Keep a **population** of networks. Each generation, the better ones are more
likely to have children; children are mixed (crossover) and slightly changed
(mutation) copies of their parents. Over generations the population drifts
toward networks that get further right.

This is the "deep GA" style of Such et al. (2017): no fancy encoding, just the
network's weights as the genome and Gaussian noise as mutation.

## What happens each generation

1. Draw 3 fresh training seeds, shared by the whole population.
2. Score all 24 networks on them. **The elites from last generation are
   re-scored too**, so a lucky score cannot keep a network on top.
3. Sort by fitness. The best becomes the current answer.
4. Build the next generation:
   - the top 2 are copied unchanged (**elitism**);
   - the other 22 are children. For each: pick a parent by **tournament**
     (3 random networks, the fittest wins); if crossover is on, pick a second
     parent the same way and take each weight from one or the other at random
     (**uniform crossover**); then add σ · N(0, 1) noise to every weight
     (**mutation**).

At 1M frames with a population of 24: about 45 generations.

## Settings

| flag | default | best found | meaning |
|---|---|---|---|
| `--pop` | 24 | 24 | population size |
| `--sigma` | 0.03 | **0.05** | mutation size |
| `--crossover` | off | **on** | uniform crossover |
| `--n-seeds` | 3 | 3 | episodes per score |
| `--stall-gens` | 0 | not yet tested | raise σ after N generations without a new best |

Tournament size (3) and elites (2) are fixed in `run_ga`.

## What we learned on Mario (validation mean over 3 runs unless noted)

| setting | validation | note |
|---|---|---|
| **σ 0.05 + crossover** | **661.8** | chosen setting; test **664.3 ± 26.0** |
| σ 0.03 + crossover | 650.7 | a statistical tie, and more consistent |
| σ 0.05 | 614.7 | |
| σ 0.03 (original) | 594.8 | |
| σ 0.03, 3M frames (1 run) | 608.4 | best score flat after ~10 generations |
| 5 seeds per score | 602.3 | fewer generations outweighed less noise |
| + previous grid + last action | 597.8 | twice the weights to search |
| + last action | 544.1 | |
| population 50 (1 run) | 539.4 | only 22 generations |
| σ 0.1 / 0.01 (1 run each) | 502.2 / 396.5 | too disruptive / too timid |

- It beats random search by ~245 points and the scripted policy (637) on average.
- Mutation size matters most; crossover helps; more budget helps little.
- Every GA model fails in the same two places: stuck at the tall pipe
  (x ≈ 595) or killed by the Goomba group after the pipes (x ≈ 700–750).

## Strengths and weaknesses

- ✅ Keeps many candidates alive: can explore several strategies at once.
- ✅ Selection only needs a *ranking*, so noisy scores hurt it less.
- ✅ Robust: every setting we tried beat random search.
- ❌ The population converges: after ~10 generations everyone is a variant of
  the same solution (see the 3M run).
- ❌ Crossover of two networks can be destructive, because the same function
  can live in different hidden units in different networks.

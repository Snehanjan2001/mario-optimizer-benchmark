# 05 · Cross-entropy method (CEM)

**Code:** [`optimizers/cem.py`](../../optimizers/cem.py) → `run_cem`
**Run:** `python -m experiments.run_cem --budget 1000000 --seed 0`

## The idea

Keep a **probability distribution** over networks — a Gaussian with a mean and
a separate spread for every weight. Sample a population from it, keep the top
20 %, and refit the Gaussian to just those. Repeat.

The distribution slides toward good regions (the mean moves) and becomes more
confident as it goes (the spread shrinks). It is one of the simplest methods
that learns *where to search*, and it is famously strong on small control
policies (Szita & Lőrincz 2006 used it to play Tetris).

## What happens each iteration

1. Draw 3 fresh training seeds.
2. Sample 24 networks: `mean + std · N(0, 1)` for every weight.
3. Score all 24.
4. Take the top 5 (20 %, the **elites**).
5. New mean = the elites' average. New std = the elites' standard deviation
   per weight, **plus a noise floor** (`extra_std = 0.02`, combined as
   √(var + 0.02²)).

That noise floor matters: 5 elites give a very small, noisy spread estimate,
and without a floor the std would collapse to near zero within a few
iterations — CEM would stop exploring before finding anything.

Generation 0 uses mean 0 and std 0.1: exactly the distribution every other
method starts from.

At 1M frames: about 45 iterations.

## Settings

| flag | default | meaning |
|---|---|---|
| `--pop` | 24 | samples per iteration |
| `--elite-frac` | 0.2 | fraction used to refit |
| `--extra-std` | 0.02 | spread floor (keeps exploring) |

The log prints the average std: watch it shrink. If it hits the floor (≈ 0.02)
early, CEM is exploring only around one point.

## CEM vs the GA vs CMA-ES

- vs **GA**: the GA keeps actual networks; CEM keeps only a distribution. The
  "child" of CEM is a sample from the averaged elites — a bit like crossover
  of all 5 elites at once.
- vs **CMA-ES**: CMA-ES refits the same kind of Gaussian but with weighted
  ranks, a learned global step size, and memory across iterations. CEM is the
  simple version.

## What to expect on Mario

- Fast early progress; the mean of 5 good networks is often good, because
  they all come from the same distribution and so share structure.
- Risk: premature convergence. Once the spread is small, CEM can only make
  tiny changes and may freeze in front of an obstacle.
- Its final answer is the **mean**, which is never itself scored during
  training.

## Strengths and weaknesses

- ✅ Very simple; few settings; strong baseline for small policies.
- ✅ Learns a per-weight spread: explores less along weights that do not matter.
- ❌ Hard top-20 % cut throws away information in the other 80 %.
- ❌ Spread estimates from 5 samples in 3,990 dimensions are very noisy.

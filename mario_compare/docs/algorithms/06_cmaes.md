# 06 · CMA-ES (sep-CMA-ES)

**Code:** [`optimizers/cmaes.py`](../../optimizers/cmaes.py) → `run_cmaes`, using the `cma` package (Hansen's reference implementation)
**Run:** `python -m experiments.run_cmaes --budget 1000000 --seed 0`

## The idea

Covariance Matrix Adaptation Evolution Strategy — the standard strong
black-box optimiser. Like CEM, it samples networks from a Gaussian and moves
the Gaussian toward the better samples. Unlike CEM it:

- uses **every** ranked sample, weighted (better ranks count more), not a hard
  top-20 % cut;
- learns the **shape** of the Gaussian: which directions have paid off, so it
  stretches the search along them;
- adapts one global **step size σ** from how consistently the mean has been
  moving (the *evolution path*): steady movement in one direction → larger
  steps; zig-zagging → smaller steps.

## Full vs diagonal (why we use sep-CMA-ES)

Full CMA-ES learns a 3,990 × 3,990 covariance matrix — 16 million numbers,
decomposed every few iterations, and it needs far more than ~45 iterations of
data to estimate. **sep-CMA-ES** (Ros & Hansen 2008) restricts it to the
diagonal: one learned spread per weight. That keeps the step-size control and
rank weighting, costs almost nothing, and is the standard choice for
neural-network weights. `--full` switches to the full matrix if you want to
try.

## What happens each iteration

1. Draw 3 fresh training seeds.
2. `ask()`: sample 24 networks from N(mean, σ² · C), where C is diagonal.
3. Score all 24.
4. `tell()`: rank them; move the mean toward a weighted average of the top
   half; update the per-weight spreads (C) and the evolution paths; update σ.
5. CMA's own early-stopping rules are switched off: the frame budget is the
   only stopping rule, as for every other method.

Starts at mean 0, σ = 0.1 — the same starting distribution as everyone else.
At 1M frames: about 45 iterations.

## Settings

| flag | default | meaning |
|---|---|---|
| `--pop` | 24 | samples per iteration (CMA's own default would be ~28 for 3,990 weights) |
| `--sigma0` | 0.1 | initial step size |
| `--full` | off | full covariance matrix |

The log prints σ each iteration: rising σ means CMA thinks it is making steady
progress; falling σ means it is homing in (or confused by noise).

## What to expect on Mario

- CMA-ES is at its best with smooth, deterministic objectives and many
  iterations. Here the objective is noisy (3 random starts per score) and
  there are only ~45 iterations, which is on the short side for CMA's
  adaptation to kick in.
- Noise tends to make σ shrink too early. If validation is poor, a larger
  `--sigma0` or more seeds per score are the first things to try.
- Its final answer is the **mean** of the distribution.

## Strengths and weaknesses

- ✅ The most principled of the distribution methods; almost no tuning.
- ✅ Invariant to how fitness is scaled (rank-based).
- ❌ Adaptation needs many iterations; 45 is few.
- ❌ Designed for deterministic problems; noise disturbs its step-size control.

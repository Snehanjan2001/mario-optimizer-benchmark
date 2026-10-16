# 01 · Random search

**Code:** [`experiments/run_baselines.py`](../../experiments/run_baselines.py) → `run_random_search`
**Run:** `python -m experiments.run_baselines --budget 1000000 --seed 0`

## The idea

Throw darts. Generate a completely random network, score it, remember it if
it is the best so far, repeat until the budget is gone. No learning, no memory
of *why* anything was good.

It is the control condition for the whole study: **anything a real optimiser
achieves beyond random search is what its machinery contributes.** The GA is
random search plus selection, mutation and crossover — the gap between them is
the honest measure of those three operators.

## What happens each iteration

1. Draw 3 fresh training seeds.
2. Draw a new weight vector: every one of the 3,990 weights from N(0, 0.1²).
3. Score it on the 3 seeds (≈ 700 frames).
4. If its fitness beats the best so far, keep it.

At 1M frames that is about 1,300 networks tried.

## Settings

| flag | default | meaning |
|---|---|---|
| `--budget` | 200000 (use 1000000) | frames |
| `--seed` | 0 | run seed |
| `--obs` | grid | network input |

The same script also runs the scripted baselines (random actions, hold right,
hold right + jump every 12/20/30/45 steps) on the test seeds first.

## What we learned on Mario

- **Test fitness 420.0 ± 69.1** (3 runs, 1M frames); validation 428.4.
- It does *worse* at 1M frames than at 50k (517). With 1,300 candidates each
  scored on only 3 seeds, the winner is usually a mediocre network that got
  lucky on its 3 starts — seed 1 found one scoring **723** on training seeds
  and **411** on test. More candidates means a luckier winner, not a better one.
  This is the *winner's curse*, and the reason every other method re-scores its
  best candidates on fresh seeds.

## Strengths and weaknesses

- ✅ Zero settings to tune, impossible to implement wrongly, embarrassingly parallel.
- ✅ The right baseline: tells you how easy the problem is.
- ❌ Ignores everything it has learned; cost of finding a good network grows
  exponentially with how rare good networks are.
- ❌ Picks winners by one noisy evaluation (winner's curse).

# The shared setup: what every algorithm gets

Every algorithm in this study plays the same game, sees the same inputs, is
scored the same way, and gets the same number of emulator frames. Only the way
it *proposes and improves networks* differs. This page describes everything
that is shared; the other pages only describe what is different.

## The game and the observation

- **Level:** Super Mario Bros 1-1, through `gym-super-mario-bros`.
- **Observation:** a 16 × 13 grid of tiles covering the screen, flattened to
  208 numbers: `0` empty, `1` solid, `2` Mario, `-1` enemy. Built from the
  game's RAM in [`envs/mario_env.py`](../../envs/mario_env.py)
  (`MarioTileEnv._build_grid`). Optional extras (`--obs act|prev|both`) add the
  last action and/or the previous grid; the default is the plain grid.
- **Actions:** 12 button combinations (`COMPLEX_MOVEMENT`: right, right+A,
  right+B, left, down, …).
- **Frame skip 4:** each decision is held for 4 emulator frames.
- **Sticky actions (p = 0.2):** at each decision, with probability 0.2 the
  game repeats the previous action instead of the chosen one
  (Machado et al. 2018). This makes every episode genuinely random, so a
  memorised button sequence cannot score well — only a real policy can.
- **Random no-op start:** 0–30 frames of doing nothing before the agent takes
  over, chosen by the episode's seed.

## The network (the thing being optimised)

Every method optimises the same network, [`policies/mlp.py`](../../policies/mlp.py):

```
208 inputs  →  18 hidden (ReLU)  →  12 outputs  →  argmax = button to press
```

That is **3,990 numbers** (weights and biases). The evolution methods treat
these as one flat vector and propose new vectors. The RL methods train the
same-shaped network with gradients, and at the end their weights are copied
into this exact layout (`rl/sb3_agents.py: export_to_mlp`), so every method is
tested with identical code.

## The score (fitness)

Per step: **+1 per pixel of new rightmost progress**. At the end: **−15 for
dying**, **+1000 for the flag**. An episode ends on death, flag, 240 frames
with no progress ("stuck"), or 3,000 decisions. Fitness = the episode total.
For reference: holding right scores 258 (stuck at the first pipe); the scripted
"hold right, jump every 20 steps" policy scores 637.

## The budget: frames, not iterations

Methods do very different amounts of work per "iteration", so they are
compared on **emulator frames used**. Every frame — including no-op starts —
is charged to one `FrameBudget` ([`core/rollout.py`](../../core/rollout.py)).
The default is **1,000,000 frames** (about 4.6 hours of game time).
Evolution methods pay through `evaluate(..., budget)`; RL pays through the
`FrameCounter` wrapper. When the budget runs out, the method stops.

## Three kinds of seeds

| seeds | used for | who sees them |
|---|---|---|
| training seeds, drawn fresh at random each iteration | scoring candidates during search | the optimiser |
| **validation**: 800000–800029 | choosing settings (σ, population, …) | you, between runs |
| **test**: 900000–900029 | the final reported number, once per chosen setting | the report |

Scoring a candidate on 3 random seeds (`--n-seeds 3`) is noisy but cheap.
All candidates in one iteration share the same seeds, so they are ranked on
equal terms.

## The end of every run

1. Take the method's final answer (its best / mean / trained network).
2. Score it on the 30 validation seeds and the 30 test seeds (not charged).
3. Log both to the run's CSV and save the weights as `<method>_best_*.npy`.

Then:

```bash
python -m experiments.score_val            # compare settings on validation
python -m experiments.watch --policy ga    # watch any method's best model play
```

## The algorithms

| page | method | family |
|---|---|---|
| [01](01_random_search.md) | Random search | baseline |
| [02](02_hill_climber.md) | (1+1) hill climber | evolution, single solution |
| [03](03_genetic_algorithm.md) | Genetic algorithm | evolution, population |
| [04](04_evolution_strategies.md) | OpenAI ES | evolution, gradient estimate |
| [05](05_cem.md) | Cross-entropy method | evolution, distribution |
| [06](06_cmaes.md) | CMA-ES | evolution, distribution + shape |
| [07](07_novelty_search.md) | Novelty search | evolution, diversity |
| [08](08_map_elites.md) | MAP-Elites | evolution, quality-diversity |
| [09](09_ppo.md) | PPO | reinforcement learning |
| [10](10_a2c.md) | A2C | reinforcement learning |
| [11](11_dqn.md) | DQN | reinforcement learning |

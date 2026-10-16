# Mario Optimiser Benchmark

A fair head-to-head comparison of optimisation methods that learn to play
**Super Mario Bros 1-1**. Eight evolution methods have been benchmarked: random
search, hill climbing, GA, ES, CEM, CMA-ES, novelty search and MAP-Elites.
Reinforcement learning methods (PPO, A2C, DQN) are implemented but have not been
run yet.

All methods share the same small network (208 → 18 → 12 MLP, 3,990 weights),
the same tile-grid observation, the same score, and the same budget of
**emulator frames**. Only the search method changes, so the comparison is
fair. The rules are in [docs/algorithms/00_setup.md](mario_compare/docs/algorithms/00_setup.md).

## Setup

Needs Python 3.13 or newer.

```bash
python3 -m venv mario-env
source mario-env/bin/activate
pip install -r requirements.txt
```

## Usage

Run everything from inside `mario_compare/`:

```bash
cd mario_compare

# 1. Baselines first: hold-right and random search
python -m experiments.run_baselines --budget 200000

# 2. Train one method (default budget is 1,000,000 frames)
python -m experiments.run_ga --budget 1000000 --seed 0
#    other methods: run_hc, run_es, run_cem, run_cmaes, run_novelty,
#                   run_mapelites, run_ppo, run_a2c, run_dqn

# 3. Compare settings on the validation seeds
python -m experiments.score_val

# 4. Watch the best model play (q = quit, space = pause, n = next episode)
python -m experiments.watch --policy ga
python -m experiments.watch --policy es --save mario.mp4 --episodes 5
```

Each script takes `--help` for its full list of options.

Each run writes a CSV log and a `<method>_best_*.npy` weight file to
`mario_compare/results/`.

## Results

Score on the 30 test seeds after 1,000,000 emulator frames, for every method
that has a finished run. Runs use seed 0. Where a method was run with several
settings, the setting with the best validation score is shown.

| Method | Test fitness (mean ± std) | Furthest x |
|---|---|---|
| CEM | **686 ± 168** | 900 |
| *Scripted: hold right, jump every 20 steps* | *637 ± 70* | *724* |
| Genetic algorithm | 627 ± 114 | 722 |
| CMA-ES | 620 ± 33 | 693 |
| Novelty search | 604 ± 40 | 693 |
| MAP-Elites | 600 ± 136 | 724 |
| Evolution strategies | 596 ± 81 | 724 |
| Hill climber | 512 ± 143 | 722 |
| Random search | 340 ± 107 | 724 |
| *Scripted: hold right* | *258 ± 1* | *315* |

**Demo:** [media/cem_demo.mp4](media/cem_demo.mp4) shows the CEM agent playing 5
test episodes. In 3 of them it reaches x=900; in the other 2 it gets stuck at x=436.

- Only CEM clearly beats the 3-line scripted jumping policy.
- No method reaches the flag within the budget.

Logs for every run are in [mario_compare/results/](mario_compare/results/).

## Project layout

```
mario_compare/
  envs/mario_env.py      Shared environment: tile-grid observation, sticky actions, our own fitness
  core/rollout.py        Shared evaluation: frame budget, seed sets, evaluate()
  policies/mlp.py        The MLP being optimised, plus scripted baselines
  optimizers/            Evolution methods (ga, es, cem, cmaes, hill_climber, novelty, map_elites)
  rl/sb3_agents.py       PPO / A2C / DQN via stable-baselines3, exported back into the MLP layout
  experiments/           One run_*.py script per method, plus score_val.py and watch.py
  docs/algorithms/       One page per method: how it works and what it changes
  results/               CSV logs and best weights from finished runs
mario_kaggle.py          Standalone script that runs the full benchmark as a Kaggle job
mario_kaggle.ipynb       The same, as notebooks for Kaggle and Colab
mario_colab.ipynb
```

## Rules that keep the comparison fair

- **Same budget:** every emulator frame is charged to one `FrameBudget`,
  including RL steps and no-op starts.
- **Randomness:** sticky actions (p = 0.2) and random no-op starts, so a
  memorised button sequence cannot score well.
- **Separate seed sets:** training seeds are drawn fresh each iteration.
  Settings are chosen on the validation seeds (800000–800029). The final
  number is reported on the test seeds (900000–900029).
- **Reference scores:** holding right scores 258. The scripted policy
  "hold right, jump every 20 steps" scores 637.

## Running on Kaggle / Colab

Upload `mario_kaggle.py` as a Kaggle script with CPU only and Internet on, then
run it. It rebuilds the project, runs every method on every seed, and writes
`results/summary.txt`. Setting `MARIO_SMOKE=1` runs a ~10-minute check
instead. Instructions are at the top of the file.

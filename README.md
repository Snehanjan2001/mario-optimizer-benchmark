# Mario Optimiser Benchmark

A fair head-to-head comparison of optimisation methods that learn to play
**Super Mario Bros 1-1**: evolution methods (random search, hill climbing, GA,
ES, CEM, CMA-ES, novelty search, MAP-Elites) and reinforcement learning methods
(PPO, A2C, DQN).

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

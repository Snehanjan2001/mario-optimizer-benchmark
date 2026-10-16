# 11 · DQN (Deep Q-Network)

**Code:** [`rl/sb3_agents.py`](../../rl/sb3_agents.py) (stable-baselines3), runner [`experiments/run_dqn.py`](../../experiments/run_dqn.py)
**Run:** `python -m experiments.run_dqn --budget 1000000 --seed 0`

## The idea

DQN (Mnih et al. 2015, the Atari paper) learns a **value** for every button
instead of a probability. The network's 12 outputs are Q-values: "if I press
this button now and play well afterwards, how much total reward will I get?".
To play, press the button with the highest Q-value.

That means the network is exactly our 208 → 18 → 12 → argmax — no separate
critic. The only difference from the evolved networks is *what the 12 outputs
mean* and how they were learned.

## What happens during training

Every step:

1. **Act (ε-greedy):** with probability ε press a random button, otherwise the
   highest-Q button. ε starts at 1.0 (pure random) and falls linearly to 0.05
   over the first 30 % of training.
2. **Remember:** store (observation, button, reward, next observation, done)
   in a **replay buffer** of the last 100,000 steps.

Every 4 steps (after the first 5,000 random ones):

3. Sample 64 random memories from the buffer and nudge the network so that
   `Q(obs, button) ≈ reward + 0.99 · max Q_target(next obs)`
   — the **Bellman equation**: the value of a press is what it earned now
   plus the best value available next.

Every 2,000 steps:

4. Copy the network into the **target network** used in step 3, so the
   target it chases does not move every step (a key stabiliser).

At 1M frames: ≈ 240,000 steps and ≈ 59,000 gradient updates.

## Why DQN is different from PPO/A2C

- **Off-policy:** it learns from old experience in the replay buffer, so every
  frame can be reused in many updates — often the most frame-efficient RL
  family on small discrete games.
- **Explicit exploration schedule:** ε, not an entropy bonus. Exploration is
  decided by the fraction of *training*, which is why `run_dqn` sets
  total steps ≈ budget / 4 so the ε schedule matches the frame budget.

## Settings

| flag | default | meaning |
|---|---|---|
| `--lr` | 1e-4 | Adam learning rate |
| `--buffer` | 100000 | replay buffer size (steps) |
| `--learning-starts` | 5000 | random steps before learning begins |
| `--target-update` | 2000 | steps between target-network copies |
| `--explore-frac` | 0.3 | fraction of training over which ε falls from 1.0 |
| `--final-eps` | 0.05 | ε after the decay |

Testing is greedy (argmax Q, ε = 0), through the same exported `MLPPolicy`
as every other method — verified identical on 100 % of test inputs.

## What to expect on Mario

- Early on it plays almost randomly (ε ≈ 1), so the first ~75,000 steps are
  mostly data collection.
- With only 18 hidden units, the network must represent the *value* of every
  button in every situation — harder than just choosing a good button, which
  is all evolution needs. Q-values for rarely-useful buttons can stay wrong
  without hurting play.
- Classic failure mode: overestimated Q-values (the `max` in the Bellman
  target is optimistic). Double DQN fixes this; SB3's DQN is the standard
  version.

## Strengths and weaknesses

- ✅ Reuses experience: potentially the most frame-efficient method here.
- ✅ No separate critic; the network is exactly the benchmark's shape.
- ❌ Many interacting settings (buffer, ε schedule, target updates).
- ❌ Learning values is harder than learning actions with a tiny network.

# 09 · PPO (Proximal Policy Optimization)

**Code:** [`rl/sb3_agents.py`](../../rl/sb3_agents.py) (stable-baselines3), runner [`experiments/run_ppo.py`](../../experiments/run_ppo.py)
**Run:** `python -m experiments.run_ppo --budget 1000000 --seed 0`

## The idea: a different family

Every method so far is **evolution**: it only sees one number per episode
("this network scored 637") and improves by trying networks. **Reinforcement
learning** instead looks *inside* episodes: after every single step it sees
the reward (pixels gained, −15 for dying) and asks "which of my button
presses led to more reward later?". It then uses backpropagation to make those
presses more likely.

PPO (Schulman et al. 2017) is the default modern RL algorithm. It is an
**actor-critic** method:

- the **actor** is the policy: our 208 → 18 → 12 network, whose 12 outputs
  become probabilities over buttons (softmax);
- the **critic** is a second network (208 → 64 → 64 → 1) that predicts "how
  much reward will I get from here?". It is only a training aid; it is never
  used to play, and it is not part of the benchmark network.

## What happens in each update cycle

1. **Collect:** play 512 steps with the current policy, *sampling* buttons
   from its probabilities (that randomness is its exploration).
2. **Advantages:** for each step, compare what actually happened afterwards
   with what the critic predicted (GAE, λ = 0.95, γ = 0.99). Positive
   advantage = "this press turned out better than expected".
3. **Learn:** 4 passes over those 512 steps in mini-batches of 64:
   - push up the probability of presses with positive advantage, down for
     negative — but **clip** the change (ratio within 1 ± 0.2) so one update
     cannot wreck the policy. That clip is the "proximal" in PPO;
   - train the critic to predict returns better;
   - add an **entropy bonus** (0.01) that rewards keeping some randomness, so
     the policy does not commit to one button too early.
4. Discard the 512 steps and repeat.

At 1M frames: ≈ 240,000 steps ≈ 470 update cycles.

## How it is made comparable

- Same env, same observation, same sticky actions, and the per-step reward is
  exactly the fitness the evolution methods are scored on, one step at a time.
- **Frames, not steps:** `FrameCounter` counts real emulator frames
  (including no-op starts) and a callback stops training when 1M is used.
- **Same network for testing:** after training, the actor's weights are copied
  into the flat `MLPPolicy` layout (`export_to_mlp`) and it plays
  *deterministically* (argmax, no sampling) on validation and test — verified
  to choose the identical button on 100 % of test inputs.
- Episodes are capped at 3,000 steps, the same cap `evaluate()` uses.

## Settings

| flag | default | meaning |
|---|---|---|
| `--lr` | 3e-4 | Adam learning rate |
| `--n-steps` | 512 | steps collected per update |
| `--batch-size` | 64 | mini-batch size |
| `--n-epochs` | 4 | passes over each batch of steps |
| `--clip` | 0.2 | how far one update may move the policy |
| `--ent-coef` | 0.01 | entropy bonus (exploration) |

The log prints the average return of recent *training* episodes every 50k
frames. Those episodes use sampled actions, so they usually score below the
deterministic test policy.

## What to expect on Mario

- RL can, in principle, learn *which step* caused a death (credit
  assignment) — information evolution never sees. That is its big advantage.
- 1M frames is small for PPO on raw-ish inputs; published Mario PPO agents
  train for tens of millions of frames with CNNs on pixels. Our tiny network
  and tile input make it much easier, but expect it to still be learning when
  the budget ends.
- Sticky actions and the random no-op start are no problem for RL: it is
  trained on a stochastic game anyway.

## Strengths and weaknesses

- ✅ Uses every step's reward, not one number per episode.
- ✅ Stable and forgiving of hyperparameters (for RL).
- ❌ Needs a second network (the critic) and many more settings than evolution.
- ❌ Gradient updates on noisy estimates; learning curves can collapse or plateau.

"""
PPO, A2C and DQN via stable-baselines3, made comparable with the evolution methods.

What is shared with every other method:
  * the same MarioTileEnv (same observation, sticky actions, termination rules)
  * the same reward: our x-progress reward, per step, instead of a per-episode total
  * the same frame budget: every emulator frame, including reset noops, is counted
  * the same network: 208 -> 18 (ReLU) -> 12, argmax at test time

What differs is the point of the comparison: RL learns from every step's reward
with gradients, while evolution only sees one number per episode.

Budget: SB3 counts agent steps, not frames. FrameCounter measures real frames
and a callback stops training the moment the frame budget is spent.

Scoring: after training, the policy's own weights (not a copy trained
separately) are exported into the flat MLPPolicy layout, so validation, test,
watch and score_val treat RL models exactly like evolved ones.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from policies.mlp import N_HIDDEN

MAX_STEPS = 3000        # same episode cap as core.rollout.rollout


class FrameCounter(gym.Wrapper):
    """Charges every emulator frame (steps and reset noops) to a FrameBudget."""

    def __init__(self, env, budget):
        super().__init__(env)
        self.budget = budget
        self._last = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.budget.used += info["frames"]          # noop frames of this reset
        self._last = info["frames"]
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.budget.used += info["frames"] - self._last
        self._last = info["frames"]
        return obs, r, term, trunc, info


class StopOnBudget(BaseCallback):
    """Ends learn() as soon as the frame budget is used up; logs progress every `every` frames."""

    def __init__(self, budget, log, method, every=50_000):
        super().__init__()
        self.budget, self.log, self.method, self.every = budget, log, method, every
        self.ep_returns, self._next = [], every

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:                    # added by SB3's Monitor wrapper
                self.ep_returns.append(info["episode"]["r"])
        if self.budget.used >= self._next and self.ep_returns:
            recent = np.array(self.ep_returns[-20:])
            self.log(dict(method=self.method, iteration=self.num_timesteps,
                          frames_used=self.budget.used, fitness=recent.mean(),
                          fitness_std=recent.std(), fitness_min=recent.min(),
                          fitness_max=recent.max()))
            print(f"  [{self.method}] steps={self.num_timesteps:7d} "
                  f"recent episode return={recent.mean():7.1f}  "
                  f"budget={self.budget.fraction:.1%}")
            self._next += self.every
        return self.budget.used < self.budget.total


# Per-algorithm defaults (SB3 / RL-zoo style). Every one can be overridden
# from that algorithm's runner: run_ppo.py, run_a2c.py, run_dqn.py.
DEFAULTS = {
    "ppo": dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01),
    "a2c": dict(learning_rate=7e-4, n_steps=16, gamma=0.99, gae_lambda=1.0, ent_coef=0.01),
    "dqn": dict(learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=2_000,
                exploration_fraction=0.3, exploration_final_eps=0.05),
}
ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def make_model(algo: str, env, seed: int, **overrides):
    """Same 18-unit ReLU policy network for every algorithm."""
    hp = {**DEFAULTS[algo], **{k: v for k, v in overrides.items() if v is not None}}
    if algo == "dqn":
        net = dict(net_arch=[N_HIDDEN], activation_fn=nn.ReLU)
    else:
        # actor gets the benchmark network; the critic (value head) is only a
        # training aid, is never used to act, and gets a normal-size network
        net = dict(net_arch=dict(pi=[N_HIDDEN], vf=[64, 64]), activation_fn=nn.ReLU)
    return ALGOS[algo]("MlpPolicy", env, seed=seed, verbose=0, policy_kwargs=net, **hp)


def export_to_mlp(model, algo: str) -> np.ndarray:
    """SB3 policy -> flat vector in MLPPolicy's layout [W1, b1, W2, b2].

    PPO/A2C: argmax over action logits. DQN: argmax over Q-values. Both are
    Linear -> ReLU -> Linear -> argmax, exactly what MLPPolicy.act computes.
    """
    if algo in ("ppo", "a2c"):
        hidden = model.policy.mlp_extractor.policy_net[0]
        out = model.policy.action_net
    else:
        hidden = model.q_net.q_net[0]
        out = model.q_net.q_net[2]
    t = lambda p: p.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([t(hidden.weight).T.ravel(), t(hidden.bias),
                           t(out.weight).T.ravel(), t(out.bias)])


def run_rl(algo: str, env, budget, log, seed=0, **hp):
    wrapped = Monitor(TimeLimit(FrameCounter(env, budget), MAX_STEPS))
    total_steps = budget.total // env.frame_skip      # upper bound; the callback stops on frames
    model = make_model(algo, wrapped, seed, **hp)
    model.learn(total_timesteps=total_steps, callback=StopOnBudget(budget, log, algo))
    return export_to_mlp(model, algo), model.num_timesteps, model

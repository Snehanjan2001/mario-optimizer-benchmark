"""
Shared evaluation protocol.

Every optimizer in the study calls evaluate() and nothing else. This is where
fairness is enforced:

  * the same seed set for every method within a generation/iteration
  * every consumed emulator frame charged to a single shared budget
  * identical episode termination rules (they live in the env)

If an optimizer ever steps the env directly, the comparison is broken.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


class BudgetExhausted(Exception):
    """Raised when the shared frame budget runs out mid-search."""


@dataclass
class FrameBudget:
    """The experimental currency. Every method gets the same total."""
    total: int
    used: int = 0

    def charge(self, frames: int) -> None:
        self.used += frames
        if self.used >= self.total:
            raise BudgetExhausted(f"{self.used}/{self.total} frames")

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def fraction(self) -> float:
        return self.used / self.total


@dataclass
class RolloutResult:
    fitness: float
    max_x: int
    frames: int
    flag: bool
    env_reward: float
    final_x: int = 0        # where the episode ended -- the "behaviour" used by
    final_y: int = 0        # novelty search and MAP-Elites. Nothing else reads it.


def rollout(env, policy, seed: int, max_steps: int = 3000) -> RolloutResult:
    """One episode. Returns our fitness, not the env's reward."""
    obs, info = env.reset(seed=seed)
    policy.reset()

    start_frames = info["frames"]
    fitness = 0.0
    env_reward = 0.0

    for _ in range(max_steps):
        obs, r, terminated, truncated, info = env.step(policy.act(obs))
        fitness += r
        env_reward += info.get("env_reward", 0.0)
        if terminated or truncated:
            break

    return RolloutResult(
        fitness=fitness,
        max_x=int(info.get("max_x", 0)),
        frames=int(info["frames"] - start_frames),
        flag=bool(info.get("flag_get", False)),
        env_reward=env_reward,
        final_x=int(info.get("x_pos", 0)),
        final_y=int(info.get("y_pos", 0)),
    )


def evaluate(env, policy, seeds, budget: FrameBudget | None = None) -> dict:
    """Average a policy over a fixed seed set.

    Noop starts make every episode stochastic, so a single rollout is a noisy
    estimate of a policy's quality. More seeds = less noise but proportionally
    more frames spent. That trade-off is itself worth a plot.
    """
    results = [rollout(env, policy, int(s)) for s in seeds]
    frames = sum(r.frames for r in results)
    if budget is not None:
        budget.charge(frames)

    fits = np.array([r.fitness for r in results], dtype=np.float64)
    return {
        "fitness": float(fits.mean()),
        "fitness_std": float(fits.std()),
        "fitness_min": float(fits.min()),
        "fitness_max": float(fits.max()),
        "max_x": float(np.mean([r.max_x for r in results])),
        "best_x": int(max(r.max_x for r in results)),
        "flag_rate": float(np.mean([r.flag for r in results])),
        "env_reward": float(np.mean([r.env_reward for r in results])),
        "frames": frames,
        "final_x": float(np.mean([r.final_x for r in results])),
        "final_y": float(np.mean([r.final_y for r in results])),
    }


def seed_set(rng: np.random.Generator, n: int, hi: int = 10**6) -> list[int]:
    """Fresh seeds for a search iteration (prevents overfitting to fixed starts)."""
    return [int(s) for s in rng.integers(0, hi, size=n)]


# Validation seeds: for choosing settings (sigma, crossover, ...) between runs.
# Never used during search, never used for final numbers.
VAL_SEEDS = list(range(800_000, 800_030))

# Held-out seeds, never used during search or for choosing settings.
# Final reported numbers use these, once, for the chosen setting only.
TEST_SEEDS = list(range(900_000, 900_030))

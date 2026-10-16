"""
Shared Mario environment for the comparative study.

Every optimizer -- GA, ES, CMA-ES, PPO, random search, scripted baseline --
talks to THIS object and nothing else. Same observation, same action space,
same fitness, same episode termination rules. That is what makes the
comparison valid.

Observation : Box(-1, 2, shape=(208,))   flattened 16x13 tile grid  (obs_mode="grid")
              optional extras, same for every method (see OBS_MODES):
                "act"  + one-hot of the action actually executed last step  (220)
                "prev" + the previous step's grid, i.e. motion              (416)
                "both" + both                                               (428)
Action      : Discrete(12)               COMPLEX_MOVEMENT
Reward      : our own, derived from info['x_pos_max'] (NOT the env's reward)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ----------------------------------------------------------------------------
# Grid geometry
# ----------------------------------------------------------------------------
GRID_W, GRID_H = 16, 13
OBS_SIZE = GRID_W * GRID_H          # 208
N_ACTIONS = 12

OBS_MODES = ("grid", "act", "prev", "both")


def obs_size(mode: str) -> int:
    return (OBS_SIZE
            + (OBS_SIZE if mode in ("prev", "both") else 0)
            + (N_ACTIONS if mode in ("act", "both") else 0))

EMPTY, SOLID, MARIO, ENEMY = 0.0, 1.0, 2.0, -1.0

# ----------------------------------------------------------------------------
# SMB ROM RAM map. These are properties of the game cartridge, not of whatever
# python wrapper you have installed, so they are stable across forks.
# ----------------------------------------------------------------------------
TILE_BASE      = 0x500   # two-page ring buffer, each page 16 wide x 13 tall
ENEMY_DRAWN    = 0x0F    # 5 slots: 0x0F .. 0x13, nonzero = enemy active
ENEMY_X_PAGE   = 0x6E    # 5 slots
ENEMY_X_SCREEN = 0x87    # 5 slots
ENEMY_Y_SCREEN = 0xCF    # 5 slots

# Calibration constants. Standing on flat ground, Mario and a walking Goomba
# should both land on row 10 -- directly on top of the floor at rows 11-12.
MARIO_Y_OFFSET = -16
ENEMY_Y_OFFSET = -24


def tile_address(tile_x: int, row: int) -> int:
    """Absolute tile column + row -> RAM address in the 0x500 ring buffer."""
    page = (tile_x // 16) % 2        # %2 because the buffer wraps every 2 pages
    return TILE_BASE + (tile_x % 16) + (page * 13 + row) * 16


def default_factory(level: str = "SuperMarioBros-1-1-v0"):
    """Builds the raw env. Edit here if your fork uses different imports."""
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    return JoypadSpace(gym_super_mario_bros.make(level), COMPLEX_MOVEMENT)


class MarioTileEnv(gym.Env):
    """Gymnasium env exposing the 208-dim tile grid."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        level: str = "SuperMarioBros-1-1-v0",
        frame_skip: int = 4,
        max_noop: int = 30,
        stall_limit: int = 240,      # frames without progress before truncating
        death_penalty: float = 15.0,
        flag_bonus: float = 1000.0,
        sticky_prob: float = 0.2,    # Machado et al. 2018 sticky actions
        obs_mode: str = "grid",
        factory=default_factory,
    ):
        super().__init__()
        self.env = factory(level)
        self.frame_skip = frame_skip
        self.max_noop = max_noop
        self.stall_limit = stall_limit
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.sticky_prob = sticky_prob
        assert obs_mode in OBS_MODES, f"obs_mode must be one of {OBS_MODES}"
        self.obs_mode = obs_mode

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_size(obs_mode),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # bookkeeping used for fitness + stall detection
        self._max_x = 0
        self._stall = 0
        self._frames = 0            # raw emulator frames consumed this episode
        self._last_frame = None
        self._prev_action = 0       # action actually executed last step
        self._prev_grid = None      # last step's grid, for obs_mode prev/both

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    @property
    def ram(self):
        return self.env.unwrapped.ram

    def _build_grid(self, info: dict) -> np.ndarray:
        ram = self.ram
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        # world-pixel x of screen column 0, straight from info
        left = int(info.get("left_x_pos", 0))

        # --- solid tiles -------------------------------------------------
        for col in range(GRID_W):
            tile_x = (left + col * 16) // 16
            for row in range(GRID_H):
                if ram[tile_address(tile_x, row)] != 0:
                    grid[row, col] = SOLID

        # --- mario -------------------------------------------------------
        m_col = (int(info.get("x_pos", 0)) - left) // 16
        m_row = (int(info.get("y_pixel", 0)) + MARIO_Y_OFFSET) // 16
        if 0 <= m_row < GRID_H and 0 <= m_col < GRID_W:
            grid[m_row, m_col] = MARIO

        # --- enemies (5 hardware slots) ----------------------------------
        for i in range(5):
            if ram[ENEMY_DRAWN + i] == 0:
                continue
            ex = int(ram[ENEMY_X_PAGE + i]) * 256 + int(ram[ENEMY_X_SCREEN + i])
            ey = int(ram[ENEMY_Y_SCREEN + i])
            e_col = (ex - left) // 16
            e_row = (ey + ENEMY_Y_OFFSET) // 16
            if 0 <= e_row < GRID_H and 0 <= e_col < GRID_W:
                grid[e_row, e_col] = ENEMY

        return grid.flatten()

    def _observe(self, info: dict) -> np.ndarray:
        grid = self._build_grid(info)
        if self.obs_mode == "grid":
            return grid
        parts = [grid]
        if self.obs_mode in ("prev", "both"):
            # first step of an episode: previous = current, i.e. "no motion"
            parts.append(grid if self._prev_grid is None else self._prev_grid)
            self._prev_grid = grid
        if self.obs_mode in ("act", "both"):
            onehot = np.zeros(N_ACTIONS, dtype=np.float32)
            onehot[self._prev_action] = 1.0
            parts.append(onehot)
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # gym api
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, info = self.env.reset(seed=seed)

        # Random no-op start. SMB is fully deterministic; without this a GA
        # can evolve a memorised button tape instead of a policy.
        # Deliberately bypasses sticky logic: prev action is 0 and the request
        # is 0, so stickiness would only burn RNG draws for no effect.
        n = int(self.np_random.integers(0, self.max_noop + 1))
        for _ in range(n):
            _, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                _, info = self.env.reset()
                break

        self._max_x = int(info.get("x_pos_max", 0))
        self._stall = 0
        self._frames = n
        self._prev_action = 0
        self._prev_grid = None
        obs = self._observe(info)
        return obs, self._pack(info)

    def step(self, action: int):
        total_env_reward = 0.0
        terminated = truncated = False
        info = {}

        # Sticky actions: one coin flip per agent decision, not per frame.
        # Uses self.np_random so the stream is reseeded by reset(seed=...).
        action = int(action)
        if self.np_random.random() < self.sticky_prob:
            action = self._prev_action
        self._prev_action = action

        for _ in range(self.frame_skip):
            _, r, terminated, truncated, info = self.env.step(int(action))
            total_env_reward += float(r)
            self._frames += 1
            if terminated or truncated:
                break

        # ---- our own reward, independent of the fork's shaped/clipped one
        x_max = int(info.get("x_pos_max", self._max_x))
        gained = max(0, x_max - self._max_x)
        self._max_x = x_max

        reward = float(gained)
        self._stall = 0 if gained > 0 else self._stall + self.frame_skip

        if info.get("flag_get"):
            reward += self.flag_bonus
            terminated = True
        if info.get("is_dead") or info.get("is_dying") or info.get("is_game_over"):
            reward -= self.death_penalty
            terminated = True
        if self._stall >= self.stall_limit:
            truncated = True

        obs = self._observe(info)
        return obs, reward, terminated, truncated, self._pack(info, total_env_reward)

    def _pack(self, info: dict, env_reward: float = 0.0) -> dict:
        out = dict(info)
        out["max_x"] = self._max_x
        out["frames"] = self._frames          # budget accounting lives here
        out["env_reward"] = env_reward        # kept for reference, never used
        return out

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ----------------------------------------------------------------------------
# Step 2b: grid calibration. Run this file directly.
# ----------------------------------------------------------------------------
SYMBOLS = {0.0: ".", 1.0: "#", 2.0: "M", -1.0: "E"}

def ascii_grid(obs: np.ndarray) -> str:
    g = obs.reshape(GRID_H, GRID_W)
    return "\n".join("".join(SYMBOLS[float(v)] for v in row) for row in g)


if __name__ == "__main__":
    env = MarioTileEnv(frame_skip=4, max_noop=0)
    obs, info = env.reset(seed=0)

    print("obs space:", env.observation_space)
    print("first info keys:", sorted(info.keys())[:8], "...")

    for t in range(120):
        obs, r, term, trunc, info = env.step(1)   # 1 = right
        if t % 20 == 0:
            ram = env.ram
            enemies = [
                (i, int(ram[ENEMY_X_PAGE + i]) * 256 + int(ram[ENEMY_X_SCREEN + i]),
                 int(ram[ENEMY_Y_SCREEN + i]))
                for i in range(5) if ram[ENEMY_DRAWN + i] != 0
            ]
            print(f"\n--- t={t} x_max={info['max_x']} frames={info['frames']} r={r}")
            print(f"    x_pos={info.get('x_pos')} left_x_pos={info.get('left_x_pos')} "
                  f"y_pixel={info.get('y_pixel')} y_pos={info.get('y_pos')} "
                  f"screen_col={(int(info.get('x_pos',0))-int(info.get('left_x_pos',0)))//16}")
            print(f"    enemies(slot,world_x,screen_y)={enemies}")
            print(ascii_grid(obs))
        if term or trunc:
            print("episode ended at t =", t, "reason:",
                  "terminated" if term else "truncated")
            break
    env.close()
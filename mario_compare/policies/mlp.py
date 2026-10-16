"""
Policy layer.

The contract every optimizer relies on:

    policy.act(obs) -> int in [0, 12)
    policy.get_params() -> flat float64 array
    policy.set_params(vec) -> None
    policy.n_params -> int

Because parameters are a flat vector, GA / ES / CMA-ES / random search are all
just different ways of proposing vectors. None of them needs to know that the
vector happens to describe a 208-18-12 network.
"""

from __future__ import annotations
import numpy as np

N_INPUTS, N_HIDDEN, N_OUTPUTS = 208, 18, 12

# COMPLEX_MOVEMENT indices (12 actions):
#  0 NOOP        1 right       2 right+A      3 right+B
#  4 right+A+B   5 A           6 left         7 left+A
#  8 left+B      9 left+A+B   10 down        11 up
RIGHT, RIGHT_JUMP, RIGHT_RUN, RIGHT_RUNJUMP = 1, 2, 3, 4


class Policy:
    """Base class. Stateless by default; reset() exists for recurrent policies."""
    n_params = 0

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def get_params(self) -> np.ndarray:
        return np.zeros(0)

    def set_params(self, vec: np.ndarray) -> None:
        pass


# ---------------------------------------------------------------------------
# The evolvable network -- same topology as the original repo
# ---------------------------------------------------------------------------
class MLPPolicy(Policy):
    """208 -> 18 (ReLU) -> 12, argmax. 3990 parameters."""

    def __init__(self, hidden: int = N_HIDDEN, init_scale: float = 0.1, rng=None,
                 n_inputs: int = N_INPUTS):
        self.hidden = hidden
        self.n_inputs = n_inputs
        self.rng = rng or np.random.default_rng()
        self.shapes = [
            (n_inputs, hidden),   # W1
            (hidden,),            # b1
            (hidden, N_OUTPUTS),  # W2
            (N_OUTPUTS,),         # b2
        ]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n_params = int(sum(self.sizes))
        self.set_params(self._random_init(init_scale))

    def _random_init(self, scale: float) -> np.ndarray:
        parts = []
        for shape in self.shapes:
            if len(shape) == 2:              # weights
                parts.append(scale * self.rng.standard_normal(shape).ravel())
            else:                            # biases start at zero
                parts.append(np.zeros(int(np.prod(shape))))
        return np.concatenate(parts)

    # -- flat vector <-> matrices -------------------------------------------
    def set_params(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64).ravel()
        assert vec.size == self.n_params, f"expected {self.n_params}, got {vec.size}"
        self._flat = vec.copy()
        out, i = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(self._flat[i:i + size].reshape(shape))
            i += size
        self.W1, self.b1, self.W2, self.b2 = out

    def get_params(self) -> np.ndarray:
        return self._flat.copy()

    # -- forward pass --------------------------------------------------------
    def act(self, obs: np.ndarray) -> int:
        h = np.maximum(0.0, obs @ self.W1 + self.b1)     # ReLU
        return int(np.argmax(h @ self.W2 + self.b2))     # argmax


# ---------------------------------------------------------------------------
# Baselines -- the two numbers every other method must beat
# ---------------------------------------------------------------------------
class HoldRightPolicy(Policy):
    """Scripted: run right, jump on a fixed duty cycle. Zero parameters.

    This is the floor. Any method that does not clearly beat it has learned
    nothing, no matter how good its fitness curve looks in isolation.
    """

    def __init__(self, jump_period: int = 0, jump_len: int = 6):
        self.jump_period = jump_period    # 0 = never jump
        self.jump_len = jump_len
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def act(self, obs: np.ndarray) -> int:
        self.t += 1
        if self.jump_period and (self.t % self.jump_period) < self.jump_len:
            return RIGHT_RUNJUMP
        return RIGHT_RUN


class RandomPolicy(Policy):
    """Uniform random actions. The absolute floor -- expect near zero."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, N_OUTPUTS))


if __name__ == "__main__":
    p = MLPPolicy()
    print("parameters:", p.n_params)          # 3990
    obs = np.zeros(N_INPUTS, dtype=np.float32)
    print("action on blank obs:", p.act(obs))
    v = p.get_params()
    p.set_params(v)                            # round-trip must be lossless
    assert np.allclose(v, p.get_params())
    print("round-trip ok")

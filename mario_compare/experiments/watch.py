"""
Watch a policy play, in a window or saved to an .mp4.

Plays episode after episode (seed, seed+1, ...) until you press q, so you can
see where it succeeds and where it dies. Each episode ends the same way it does
in training: death, flag, or 240 frames without progress ("stuck").

Usage:
    python -m experiments.watch                                  # best GA model by held-out score
    python -m experiments.watch --policy es                      # best ES model
    python -m experiments.watch --policy cem --by val            # best CEM model by validation score
    python -m experiments.watch --policy ppo                     # ... or hc, cem, cmaes,
                                                                 #     novelty, mapelites, a2c, dqn
    python -m experiments.watch --weights results/ga_best_X.npy --seed 900003
    python -m experiments.watch --policy jump20                  # scripted baseline
    python -m experiments.watch --episodes 1                     # one episode, then close
    python -m experiments.watch --save mario.mp4 --episodes 5    # write video, no window

Keys in the window: q = quit, space = pause, n = skip to next episode.
"""

from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.mario_env import MarioTileEnv, obs_size
from policies.mlp import MLPPolicy, HoldRightPolicy

RESULTS = Path(__file__).resolve().parents[1] / "results"
SCALE = 3
END_HOLD_S = 2.0        # how long the last frame stays up with the end reason


def best_weights(method: str, by: str = "test") -> Path:
    """The <method>_best_*.npy whose run scored highest.

    by="test": the held-out score logged at the end of each run.
    by="val":  the validation score (runs from before validation existed are skipped).
    """
    row_name = f"{method}_heldout" if by == "test" else f"{method}_val"
    scored = []
    for f in RESULTS.glob(f"{method}_*.csv"):
        npy = RESULTS / f.name.replace(f"{method}_", f"{method}_best_", 1).replace(".csv", ".npy")
        rows = [r for r in csv.DictReader(f.open()) if r["method"] == row_name]
        if rows and npy.exists():
            scored.append((float(rows[-1]["fitness"]), npy))
    if not scored:
        sys.exit(f"no finished {method.upper()} run with a {by} score and saved weights in results/")
    fit, path = max(scored)
    label = "held-out" if by == "test" else "validation"
    print(f"best {method.upper()} model by {label}: {path.name}  ({label} fitness {fit:.1f})")
    return path


def obs_mode_of(path: Path) -> str:
    """Input type is in the file name (_obsact, _obsprev, _obsboth); none = grid."""
    m = re.search(r"_obs([a-z]+)", path.stem)
    return m.group(1) if m else "grid"


def load_policy(args):
    """Returns (policy, name, obs_mode)."""
    if args.policy == "jump20":
        return HoldRightPolicy(jump_period=20), "hold_right_jump20", "grid"
    path = Path(args.weights) if args.weights else best_weights(args.policy, args.by)
    mode = obs_mode_of(path)
    policy = MLPPolicy(n_inputs=obs_size(mode))
    policy.set_params(np.load(path))
    return policy, path.name, mode


def end_reason(info, term, trunc):
    if info.get("flag_get"):
        return "FLAG!"
    if term:
        return "died"
    if trunc:
        return "stuck (no progress)"
    return "quit"


def draw(env, lines):
    frame = cv2.cvtColor(env.env.unwrapped.screen, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_NEAREST)
    # dark panel so the overlay stays readable over the game's own HUD
    panel = frame[:12 + 32 * len(lines), :330]
    panel[:] = (panel * 0.35).astype(frame.dtype)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help=".npy from a GA / random-search run")
    ap.add_argument("--policy", default="ga",
                    choices=["ga", "es", "hc", "cem", "cmaes", "novelty", "mapelites",
                             "ppo", "a2c", "dqn", "jump20"],
                    help="which best model to load when --weights is not given")
    ap.add_argument("--by", choices=["test", "val"], default="test",
                    help="pick the best model by held-out (test) or validation score")
    ap.add_argument("--seed", type=int, default=900_000, help="first episode's seed")
    ap.add_argument("--episodes", type=int, default=0, help="0 = until you press q")
    ap.add_argument("--sticky-prob", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=15, help="agent steps per second")
    ap.add_argument("--save", help="write an .mp4 here instead of opening a window")
    args = ap.parse_args()
    if args.save and args.episodes == 0:
        args.episodes = 5           # a video needs an end

    policy, name, mode = load_policy(args)
    env = MarioTileEnv(sticky_prob=args.sticky_prob, obs_mode=mode)
    title = f"Mario - {name}"
    writer = None

    def show(frame, wait_ms):
        """Returns the key pressed (window) or None (video)."""
        nonlocal writer
        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (w, h))
            for _ in range(max(1, wait_ms * args.fps // 1000)):
                writer.write(frame)
            return None
        cv2.imshow(title, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    ep, quit_all, xs = 0, False, []
    while not quit_all and (args.episodes == 0 or ep < args.episodes):
        seed = args.seed + ep
        obs, info = env.reset(seed=seed)
        policy.reset()
        fitness, term, trunc, paused, skip = 0.0, False, False, False, False

        while not (term or trunc):
            if not paused:
                obs, r, term, trunc, info = env.step(policy.act(obs))
                fitness += r
            key = show(draw(env, [f"ep {ep + 1}  seed {seed}",
                                  f"x={info['max_x']}  fitness={fitness:.0f}"]),
                       1000 // args.fps)
            if key == ord("q"):
                quit_all = True
                break
            if key == ord("n"):
                skip = True
                break
            if key == ord(" "):
                paused = not paused

        reason = end_reason(info, term, trunc) if not skip else "skipped"
        xs.append(info["max_x"])
        print(f"ep {ep + 1:3d}  seed={seed}  x={info['max_x']:5d}  "
              f"fitness={fitness:7.1f}  {reason}")
        if not quit_all:
            key = show(draw(env, [f"ep {ep + 1}  seed {seed}",
                                  f"x={info['max_x']}  fitness={fitness:.0f}",
                                  f"END: {reason}"]),
                       int(END_HOLD_S * 1000))
            quit_all = key == ord("q")
        ep += 1

    print(f"{name}: {len(xs)} episodes, mean x={np.mean(xs):.1f}, best x={max(xs)}")
    if writer is not None:
        writer.release()
        print(f"wrote {args.save}")
    else:
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()

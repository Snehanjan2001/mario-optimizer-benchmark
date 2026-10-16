# 08 · MAP-Elites

**Code:** [`optimizers/map_elites.py`](../../optimizers/map_elites.py) → `run_map_elites`
**Run:** `python -m experiments.run_mapelites --budget 1000000 --seed 0`

## The idea

Instead of one population chasing one score, keep a **map**: a grid where
each cell is a *kind* of behaviour, and each cell holds the best network found
for that kind (its **elite**). Mouret & Clune (2015) called this
*illuminating* the search space: the result is a whole collection of
different, locally-best solutions, not a single one.

For Mario, a cell is **where the episode ended**:

- a 64-pixel column of the level (final x // 64), and
- a 32-pixel height band (final y // 32).

So "died in the first pit", "stuck on top of the tall pipe" and "reached the
second Goomba group" are separate cells, and each keeps its own champion.

## What happens each iteration

1. Draw 3 fresh training seeds.
2. First iteration only: 24 random networks (same init as every method).
   After that: 24 children, each made by picking a random champion from the
   map (uniformly over occupied cells), crossing it with another random
   champion, and adding σ · N(0, 1) noise.
3. Score each child and work out its cell.
4. If the cell is empty, or the child beats that cell's champion, the child
   becomes the champion.
5. The final answer is the fittest champion on the whole map.

Because parents are picked *uniformly over cells*, a rare behaviour (say,
landing on the tall pipe) gets as many children as the common "stuck in front
of the pipe" behaviour. That is how MAP-Elites keeps stepping stones alive.

## Settings

| flag | default | meaning |
|---|---|---|
| `--batch` | 24 | children per iteration |
| `--sigma` | 0.05 | mutation size |
| `--no-crossover` | off | crossover between champions is on by default |

Fixed in code: 64-px x bins, 32-px y bins (≈ 50 × 8 possible cells on 1-1).
The log prints how many cells are filled and how many children were placed.

## A known weakness here: noise

A champion's score comes from **one** evaluation on 3 seeds and is never
re-checked. A lucky child can take a cell and hold it forever, blocking
genuinely better networks. The GA avoids this by re-scoring elites every
generation; MAP-Elites (in its basic form) does not. Expect a noticeable gap
between its training scores and its validation/test scores — the validation
seeds will show exactly how much.

## Strengths and weaknesses

- ✅ Keeps many different strategies alive at once; excellent at escaping
  dead ends in principle.
- ✅ The map itself is informative: it shows where along the level networks
  tend to end up.
- ❌ Spends much of the budget improving cells that do not matter for the
  score (e.g. "died early, but in a new way").
- ❌ Sensitive to noisy evaluation (see above).

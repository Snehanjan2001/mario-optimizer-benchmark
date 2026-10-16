# 07 · Novelty search (with fitness)

**Code:** [`optimizers/novelty.py`](../../optimizers/novelty.py) → `run_novelty`
**Run:** `python -m experiments.run_novelty --budget 1000000 --seed 0`

## The idea

A GA that also rewards **doing something new**, not only going further right.

Mario 1-1 is *deceptive*: to get past the tall pipe you may first need
networks that jump higher, even if jumping higher does not yet score more.
When the whole population is stuck at the same pipe, they all score ≈ 555 and
plain selection has nothing to choose between. Novelty search (Lehman &
Stanley 2011) says: among those, prefer the ones that ended up somewhere
unusual — they are stepping stones.

## Behaviour and novelty

- **Behaviour** of a network = where its episodes ended: (final x, final y),
  in tiles, averaged over the evaluation seeds. "Died in the pit", "stuck on
  top of the pipe" and "stuck in front of the pipe" are different behaviours
  even if their fitness is similar.
- **Novelty** = the average distance from a network's behaviour to its 10
  nearest neighbours among the current population and an **archive** of past
  behaviours. Far from everything seen before → novel.
- The archive gets a random 10 % of each generation's behaviours, so it
  remembers where the search has already been.

## What happens each generation

1. Draw 3 fresh seeds; score all 24 networks; record fitness **and** behaviour.
2. Compute novelty for each network.
3. **Selection score** = 0.5 · rank(novelty) + 0.5 · rank(fitness).
   (`--novelty-weight 1` is pure novelty search; `0` is the plain GA.)
4. Next generation: the top 2 **by fitness** are kept (elitism); the other 22
   are children of tournament winners **by selection score**, with uniform
   crossover and Gaussian mutation, exactly as in the GA.
5. The returned network is the fittest of the last full generation, so the
   benchmark measures the same thing as for every other method.

## Settings

| flag | default | meaning |
|---|---|---|
| `--novelty-weight` | 0.5 | balance between novelty and fitness in selection |
| `--sigma` | 0.05 | mutation size (the GA's best) |
| `--pop` | 24 | population |
| `--no-crossover` | off | crossover is on by default, as in the GA's best setting |

Fixed in code: 10 nearest neighbours, 10 % archive rate, behaviour in tiles.

## What to expect on Mario

- Compared with the GA at the same settings, this tests one question: **does
  rewarding diversity help escape the pipe and Goomba dead ends?**
- Pure novelty (`--novelty-weight 1`) tends to wander: it happily explores
  going left, or dying in unusual places. The blend keeps it pointed right.
- Final y is a crude behaviour; a richer one (e.g. the path, not just the end
  point) would make "different" more meaningful, at the cost of complexity.

## Strengths and weaknesses

- ✅ Directly targets deceptive problems and plateaus.
- ✅ Same machinery and settings as the GA, so the comparison is clean.
- ❌ Depends entirely on a good behaviour definition.
- ❌ Spends some of the budget exploring behaviours that never pay off.

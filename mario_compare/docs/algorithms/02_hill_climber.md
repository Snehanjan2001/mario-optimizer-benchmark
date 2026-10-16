# 02 · (1+1) hill climber

**Code:** [`optimizers/hill_climber.py`](../../optimizers/hill_climber.py) → `run_hill_climber`
**Run:** `python -m experiments.run_hc --budget 1000000 --seed 0`

## The idea

Keep **one** network (the parent). Make one slightly changed copy (the child).
Keep whichever is better. Repeat.

"(1+1)" is evolution-strategies notation: 1 parent, 1 child, and the next
parent is chosen from both. It is the smallest possible evolutionary
algorithm — a GA with a population of one and no crossover.

**Why it is in the benchmark:** it isolates what the GA's *population* adds.
If the GA barely beats the hill climber, the population is not what makes the
GA work; if it beats it clearly, having many candidates at once matters.

## What happens each iteration

1. Draw 3 fresh training seeds.
2. Child = parent + σ · noise, with noise ~ N(0, 1) for every weight (σ = 0.05).
3. Score **both** parent and child on those same 3 seeds.
4. If child ≥ parent, the child becomes the new parent.

Two design choices worth understanding:

- **The parent is re-scored every iteration.** That doubles the cost, but
  otherwise a parent that got one lucky score (the winner's curse from
  random search) would block every child forever.
- **Ties go to the child (`>=`).** When every variant is stuck at the same
  pipe, all score 258. Accepting equal children lets the parent *drift*
  across this flat region until a change happens to clear the pipe.
  With `>` it would freeze.

At 1M frames: about 700 iterations, i.e. 700 small steps.

## Settings

| flag | default | meaning |
|---|---|---|
| `--sigma` | 0.05 | mutation size (same as the GA's best) |
| `--n-seeds` | 3 | episodes per score |

The log prints the **acceptance rate**. Very high (> 60 %) means most changes
are neutral — σ could be larger. Very low (< 5 %) means the climber is stuck
or σ is too large.

## What to expect on Mario

- Fast, steady improvement early; then it tends to get stuck at the first
  obstacle it cannot solve with one small change (the tall pipe, the Goomba
  group). There is no second candidate exploring elsewhere.
- Its final answer is the **parent**, so what gets tested is a network that
  has survived many head-to-head re-scorings — usually a reliable one.

## Strengths and weaknesses

- ✅ Simplest real optimiser; one setting (σ).
- ✅ Every frame goes to improving one solution.
- ❌ One path through the search space: dead ends are fatal.
- ❌ Half the budget is spent re-scoring the parent.

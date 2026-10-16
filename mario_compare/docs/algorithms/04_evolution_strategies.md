# 04 · Evolution strategies (OpenAI-ES)

**Code:** [`optimizers/es.py`](../../optimizers/es.py) → `run_es`
**Run:** `python -m experiments.run_es --budget 1000000 --seed 0`

## The idea

Keep **one** network, θ. Instead of keeping good candidates (like the GA), use
candidates only as *measurements*: try many small random changes to θ, see
which ones scored better, and move θ in the direction they point.

It is gradient ascent where the gradient is **estimated by trial and error**
instead of computed by backpropagation (Salimans et al. 2017). Mathematically
it follows the gradient of the *average* fitness of a Gaussian cloud of
networks around θ:

```
∇ ≈  1 / (n σ)  ·  Σᵢ  score(θ + σ εᵢ) · εᵢ
```

## What happens each iteration

1. Draw 3 fresh training seeds, shared by every candidate.
2. Draw 12 noise vectors ε₁…ε₁₂, each with 3,990 entries ~ N(0, 1).
3. Score **θ + σε** and **θ − σε** for each (24 candidates). Testing both
   signs (**antithetic sampling**) cancels much of the noise: if + beats −,
   that direction is uphill.
4. Replace the 24 scores by their **centred ranks** in [−0.5, 0.5]. Only the
   ordering matters, so one lucky 900-point episode cannot hijack the update.
   Tied scores are put in **random order**: on a plateau (every candidate
   stuck at the same pipe) the update becomes a random step, so θ
   random-walks until something scores differently. The reference
   implementation gets the same effect from argsort's arbitrary order.
5. Gradient estimate: for each pair, (rank of +) − (rank of −), times its ε,
   summed and divided by 2 · 12 · σ.
6. Step θ with **Adam** (learning rate 0.01), plus a tiny weight decay applied
   outside Adam.
7. All 24 candidates are thrown away. Only θ survives.

At 1M frames: about 45 iterations — 45 gradient steps.

## Settings

| flag | default | meaning |
|---|---|---|
| `--pairs` | 12 | antithetic pairs per iteration (24 candidates) |
| `--sigma` | 0.02 | size of the probing noise |
| `--lr` | 0.01 | Adam step size |
| `--n-seeds` | 3 | episodes per score |

σ controls how far the probes reach (too small: all probes score the same;
too large: the gradient estimate stops meaning anything). The learning rate
controls how far θ moves on each estimate.

## What to expect on Mario

- ES is designed for **thousands** of iterations with **thousands** of probes
  (the original paper used 1,440 CPUs). At 1M frames it gets ~45 noisy steps.
  It may well lose to the GA here — that is a real finding about small budgets:
  *keeping* the best candidates (GA) is more frame-efficient than *estimating a
  direction* from them (ES).
- Its answer is θ itself, which is never scored during training — only its
  perturbations are. Validation/test scores are the first real measurement.

## Results so far (1M frames, 3 runs each)

| setting | validation per run | validation mean | test mean |
|---|---|---|---|
| `--sigma 0.05 --lr 0.03` | 611, 242, 494 | **449.1** | 449.2 ± 150.6 |
| default (σ 0.02, lr 0.01) | 519, **0**, 377 | 298.7 | 303.6 ± 225.2 |

Both seed-1 runs froze, which explains the huge spread:

- **default, seed 1:** the random starting network never moves Mario. All 24
  perturbations (σ = 0.02 is small) also scored 0, from the first iteration
  to the last.
- **σ 0.05, seed 1:** reached 242 in 8 iterations, then every candidate tied
  at ≈ 242 for the remaining 75 iterations.

These runs used an earlier version that *averaged* tied ranks, which gives a
zero update on a plateau — ES could never move again. Ties are now broken at
random (above). **Re-run ES before trusting these numbers.** A re-test of the
seed-1 start still sat at 0 after 8 iterations, though: ES searching from a
*single* starting point is a real weakness here. The GA starts from 24
different random networks, so one bad start does not sink the run.

## Strengths and weaknesses

- ✅ Scales almost perfectly across machines (each probe is independent).
- ✅ Rank-based, antithetic: robust to noisy and oddly-scaled rewards.
- ✅ Smooth, gradual improvement of a single solution.
- ❌ Needs many iterations; throws away every evaluated candidate.
- ❌ Estimates the *average* fitness around θ, so it prefers robust regions
  and can miss narrow, precise solutions (like an exact jump timing).

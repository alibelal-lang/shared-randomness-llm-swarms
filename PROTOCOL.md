# Confirmatory protocol (locked 7 Nov 2026, before any confirmatory run)

Written after calibration only: Qwen 3.6 35B via OpenRouter, seeds 10000+ (see `results/CALIBRATION_NOTES.md`).
No Adaption-hosted policy has been simulated when this file was written. Seeds 0-499 have not been used.

## Models
All Adaption-hosted models that answer with a bare letter when `reasoning.effort = "none"`:
`qwen3.8-27b`, `qwen3.5-397b`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `glm-5.2`.
Every model is reported. **Primary model: `qwen3.8-27b`**, chosen for access before its full policy was seen.
It is also served on OpenRouter with logprobs, which allows an exact-probability cross-check.

Competence gate, decided now: a variant is analysed only if invalid answers are under 10% of samples.
Failing variants are reported as failures, not dropped silently.

## Policy extraction (Adaption)
Each of the 84 prompts (21 histories x 2 framings x 2 label mappings, H = 2) is sampled 200 times at temperature 1.0.
Calls are stateless: `store=False`, `agentic_memory=False`, no stack. The two-label policy is estimated as
(k_A + 0.5) / (k_A + k_B + 1). Invalid answers are counted and reported.

## Simulation design
N in {4, 8, 16, 32}; T = 100 rounds; 500 population runs per cell (seeds 0-499), paired across samplers.
Samplers: `independent`, `pair_shared:restore` (primary intervention, variance restored to the N0 = 4 reference),
`pair_shared:0.3` (fixed-alpha sensitivity), `block:2` (equal-variance comparator with thinner tails).
Primary start: `incumbent_B`. Robustness starts: `empty`, `split`.

## Endpoints
- **Primary endpoint:** late-window mean share playing A (rounds 51-100) of each population run.
  The x >= 0.75 entry endpoint was saturated in calibration, so it is demoted to secondary.
- **Primary contrast (per model):** at N = 32, `pair_shared:restore` minus `independent` in the primary endpoint,
  averaged with equal weight over the four prompt variants.
- **Secondary:**
  - the same contrast for each variant separately;
  - the scale interaction (the contrast at N = 32 minus the contrast at N = 4);
  - the independent-sampling N effect (N = 32 minus N = 4);
  - mean pairwise reward;
  - entry into x >= 0.9 by T;
  - share of runs at x >= 0.75 in the final round;
  - `block:2` vs `pair_shared:restore` (variance vs tails).
- **Diagnostic only:** reaching all-A, because unanimity is exponentially rare under independence by construction.

## Uncertainty
- **Layer 1, population runs:** seed-paired differences with normal 95% intervals.
- **Layer 2, policy estimation:** parametric bootstrap. Each replicate draws every history's p from
  Beta(k_A + 0.5, k_B + 0.5), reruns 100 fresh seeds (100000+) per cell, and recomputes the contrast.
  B = 100 replicates. The reported interval is the 2.5-97.5 percentiles over replicates.

## Decision rule
An exploration effect is claimed for a model only if the layer-2 95% interval for its primary contrast excludes 0.
The claim is limited to that model, the two-label constrained policy and this finite-memory protocol.
Null and opposite-sign results are reported with their intervals.

## Live validation (Adaption, independent arm only)
Primary model, all four variants, N in {4, 16}, 20 runs x 30 rounds, with the same matching seeds as simulation.
Each decision is one stateless Adaption call. An invalid answer is retried up to 3 times. If it is still invalid,
the action is drawn uniformly at random and logged; this rule is the same for every condition.
Agreement is checked in two ways: per-decision calibration of live actions against the extracted p, and live vs
simulated mean x per round. The coupled arms cannot be run live on a hosted API, because the hosted API draws its
own randomness; they remain interventions on the extracted policy.

## Amendment 1 (7 Nov 2026, 14:35, before any Adaption policy or live data existed)
Adaption enforces a per-user inference rate limit (HTTP 429, retry-after 60 s; observed ceiling about 10 calls/s
across all models). To fit the sprint:
- `qwen3.8-27b` (primary) keeps 200 samples per prompt. Secondary models use 100 samples per prompt.
- Live validation covers variants f0m0 and f1m1 (one per framing, both label mappings), not all four.
  N in {4, 16}, 20 runs x 30 rounds are unchanged.
- Calls go through a shared adaptive rate limiter. A 429 is waited out and is never counted as an invalid answer.

## Learning pilot (Adaption AutoScientist), exploratory
Base `Qwen/Qwen3.5-4B`. Reward-filtered SFT on decisions from frozen-policy rollouts, with one training dataset
from independent-sampler rollouts and one from coupled rollouts at N = 32, matched row counts. Adapters are
evaluated locally on the 21 histories and resimulated. This is offline reward-filtered fine-tuning, not on-policy RL,
with one training run per arm. It is a feasibility pilot, not a confirmatory test.

## Amendment 3 (7 Nov 2026, 21:20 UTC): extended study
Written after the scoping results and after the provider replication in B, before any exploration-grid, temperature,
memory, stag-hunt or new-wording data existed, and before the Adaption retests finished. The original confirmatory
results stand unchanged. Everything below is reported as an extension. Exploration uses exact probabilities from
OpenRouter; confirmation uses Adaption sampling.

### A. Precision and retest (Adaption)
- Fresh extraction of the original 84 prompts: `qwen3.8-27b` at 300 samples and `glm-5.2` at 400 samples.
- `glm-5.2`, variant f0m1, histories AA.AA and BB.AA: 5,000 extra samples each. Both were 100/100, and GLM's
  scoping effect moves from +4 to +23 points depending on the unmeasured tail.
- Refill samples lost to API errors (`qwen3.5-397b`: 489 across 9 prompts; `gemma-4-31b-it`: 56 across 4).
- Retest rule: for each prompt, z = difference in P(A) between fresh and original over its pooled standard error.
  If the sum of z^2 over the 84 prompts is below the chi-square(84) 99th percentile (about 117.6), pool all samples.
  Otherwise report drift and use the fresh extraction alone.
- The confirmatory simulations and layer-2 bootstrap are rerun on the resulting policy and reported beside the
  original numbers.

### B. Deployment (OpenRouter, already run)
- The original 84 prompts on 15 provider deployments (5 for Qwen 3.8, 4 GLM, 1 Qwen 397B, 2 Gemma 31B, 3 Gemma 26B).
  One call per prompt, exact top-20 logprobs, reasoning disabled.
- Validity gate: J+K probability mass at least 0.9 in every variant (`glm-5.2` on morph-fp8 fails at 0.82).
- Pinned deployment per model, by a rule written before any effect was computed: the valid provider with the smallest
  mean |p - p_Adaption|. Pins: wafer-fp8 (Qwen 3.8), ambient-fp8 (GLM), parasail-fp8 (Qwen 397B),
  novita-bf16 (Gemma 31B), coreweave-bf16 (Gemma 26B).
- Seen already, and reported as a finding: the same weights served by different providers give different scoping
  effects. For example, Qwen 3.8's coupling effect ranges from +0.4 to +3.5 points across five providers, with the
  same sign on all five.

### C. Exploration grid (OpenRouter, exact probabilities)
- Wording: framings 0-7. 0-1 are the originals; 2-7 are new and semantically equivalent (formal payoff table;
  strangers and dollars; AI agents and protocols; terse; per-option description; business standards).
- Labels: mappings 0-3, crossing which letter is the better option with which letter is listed first.
- Game: `coord` (original) and `stag` (A-A 4, B-B 3, B pays 3 whatever the partner does, A alone pays 0).
  B is risk-dominant in `stag`, so the incumbent is stable for a reason, not only by habit.
- Memory: H = 2 for the full grid on every valid deployment from B. H = 1 and H = 3 on framings 0-1 x mappings 0-3
  x both games, pinned deployments only.
- Temperature: T in {0.5, 0.75, 1, 1.5}, applied to the exact two-label logits (p_T = sigmoid(logit / T)),
  pinned deployments, H = 2.
- Simulation per prompt variant: seeds 0-199, T = 100 rounds, incumbent_B start, late-window share of A.
  Cells: independent at N = 4, 32, 64 and pair_shared:restore (N0 = 4) at N = 32, 64.
- Readouts: size effect (independent, N = 32 minus N = 4) and coupling effect (restore minus independent at N = 32),
  plus the same at N = 64.
- A variant is **responsive** if its size effect at N = 32 is -2 points or lower.

### D. Adaption confirmation queue (in this order, until 8 Nov 18:00 UTC)
1. `stag` on the original prompt set (framings 0-1 x mappings 0-1) for `qwen3.8-27b` and `glm-5.2`, 200 samples.
   Run whatever the exploration shows.
2. New wordings: for each of `qwen3.8-27b` and `glm-5.2`, up to 6 responsive H = 2 variants (framing, mapping, game)
   on the pinned deployment, choosing the most negative size effect first and excluding variants already queued.
   200 samples.
3. Temperature: `qwen3.8-27b`, original prompt set, sampled at temperature 0.7 (200 samples). This is compared with
   the prediction from rescaling the temperature-1 Adaption policy.
4. Memory: H = 3 on the original prompt set (100 samples) for the model and game with the most responsive H = 3
   variants on OpenRouter (ties go to `qwen3.8-27b` and `coord`).
5. `qwen3.5-397b`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`: `stag` on the original prompt set (100 samples), only if the
   model has at least one responsive `stag` variant on OpenRouter.

### E. Confirmation analysis
- Same as the original protocol: seeds 0-499; N in {4, 8, 16, 32, 64}; independent, pair_shared:restore,
  pair_shared:0.3 and block:2. Layer-2 bootstrap (B = 100) on the N = 32 vs 4 contrasts.
- Coupling dose-response: pair_shared with alpha in {0.1, 0.2, 0.3, 0.5, 0.75, 1} at N = 32 and 64.
- Each confirmed set is reported with its interval, whatever the outcome. The family summary is the number of sets
  whose interval excludes 0, and their signs.

### F. Expectations stated in advance
- Mechanism: coupling is positive where the size effect is negative and about 0 elsewhere, recovering between 0% and
  100% of the size penalty.
- Scale: the size penalty and the coupling effect are at least as large at N = 64 as at N = 32.
- Stag hunt: more responsive variants and larger coupling effects than `coord`.
- Dose-response: the coupling effect rises with alpha.
- Temperature, letter and list order: no direction predicted. Effects are reported separately for letter and order.

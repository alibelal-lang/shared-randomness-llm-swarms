# Calibration notes (7 Nov 2026)

These are calibration and pilot results, run before any protocol was locked. Confirmatory runs should
use fresh seeds after the endpoint, threshold and horizon are fixed.

## Sampler validation
- The exact reference table in the research plan reproduces to the stated precision for all 12 cells
  (`python -m swarm reference`; `tests/test_core.py::test_reference_table_matches_plan`).
- Monte Carlo checks confirm individual marginals are preserved for every sampler, within-pair joint
  distributions are preserved for `pair_shared` and `block`, and state-calibrated `pair_shared:restore`
  hits the N0 = 4 reference variance on heterogeneous states.

## Model screening (OpenRouter, framing 0, mapping 0, 21 histories, 1 repeat)

| Model | Label mass on J/K | Behaviour | Status |
|---|---|---|---|
| qwen/qwen3.6-35b-a3b (CoreWeave, pinned) | 0.94-1.00 | stochastic, history-sensitive | **primary candidate** |
| google/gemma-4-26b-a4b-it (4 providers, unpinned) | 1.00 | P(A) ~ 1 for every history: no lock-in | screened; evidence against lock-in |
| google/gemma-3-27b-it | mean 0.68, min 0.00 | near-deterministic history copying | screened; competence issue |
| mistralai/mistral-small-3.2-24b-instruct | ~0 | does not answer with a bare letter | excluded |
| meta-llama/llama-3.2-3b-instruct (Parasail) | ~0.4 | prefers K even after J-J success; Cloudflare returned no logprobs | excluded (file not kept) |

## Qwen 3.6 35B policy (T = 1, 2 repeats)
- Repeat-to-repeat spread of P(A) reaches 0.11 on one history even with a pinned provider. Average more
  repeats or report this as extraction noise.
- Large label-mapping and framing effects. P(A | BB.BB): f0m0 0.25, f0m1 0.62, f1m0 0.25, f1m1 0.62.

## Calibration sweep (incumbent_B start, 200 runs per cell, seeds 10000+, T = 100)
- Entry into x >= 0.75 happens in every run at every N, so that endpoint is saturated: 0.75 sits near the
  independent-sampling stationary share. Choose a different primary endpoint before confirmation.
- Only f0m0 shows a population-size effect. Under independent sampling the late-window mean share of A
  falls from 0.81 (N = 4) to 0.69 (N = 32), and mean reward from 1.37 to 1.03. Under `pair_shared:restore`
  they stay at about 0.81 and 1.35 for every N. `block:2` (equal variance, thinner tails) restores less (0.75 at N = 32).
- f0m1: immediate adoption of A at every N (no trap). f1m0: stays near x = 0.38. f1m1: x = 0.73.
  None of the three show an N effect on mean share or reward.
- All-A reachability changes with N under independent sampling in every trapped variant. Unanimity is
  exponentially rare under independence by construction, so treat it as a diagnostic, not the claim.
- Working mechanism hypothesis: the effect needs a near-absorbing better state (P(A | AA.AA) = 0.986 in
  f0m0 vs 0.71-0.83 elsewhere) that coordinated cross-pair excursions can reach.

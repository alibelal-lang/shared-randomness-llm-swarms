# Findings (7 Nov 2026)

Report page: https://claude.ai/artifact/7oPtHLKXQJgcPTxUVqAANZ (private until shared). Protocol: `PROTOCOL.md` (locked before confirmatory runs, two amendments).
All LLM decisions came from Adaption-hosted models. Fine-tuning used Adaption AutoScientist.

## Primary result (frozen measured policies, 32 vs 4 agents, 500 runs per cell, layer-2 bootstrap B = 100)

Coupling effect: the contrast of `pair_shared:restore` minus independent sampling at N = 32. Size effect: independent sampling, N = 32 minus N = 4.
Both are in points of the long-run share choosing A, rounds 51-100, averaged over four prompt variants.

| Model (Adaption) | Coupling effect | Size effect | Reward gain | Decision |
|---|---|---|---|---|
| Qwen 3.8 27B (primary) | +1.5 [+0.5, +2.7] | -1.4 [-2.5, -0.7] | +0.067 | effect claimed |
| GLM 5.2 | +3.0 [+0.8, +6.3] | -7.3 [-9.6, -2.7] | +0.082 | effect claimed |
| Qwen 3.5 397B | -0.2 [-0.9, +0.2] | +1.0 [-0.1, +5.8] | -0.012 | none (settles at about 99% A) |
| Gemma 4 31B | 0.0 [-0.1, +0.1] | 0.0 | 0.000 | none (always chooses A) |
| Gemma 4 26B A4B | 0.0 [-0.1, +0.1] | 0.0 | 0.000 | none (always chooses A) |

- Effects sit in single prompt variants: Qwen 3.8 in "framing 1, J better" (+4.0) and GLM 5.2 in "framing 1, K better" (+12.1).
  In GLM's variant, independent populations drop from 0.87 (N = 4) to 0.61 (N = 32).
- In both models coupling roughly cancels the size penalty. The scale interaction excludes zero for both.
- Variance vs tails: for Qwen 3.8 the equal-variance block comparator recovers less than the coupled sampler (+0.5 points, CI excludes 0).
  For GLM 5.2 they recover the same amount.
- Initial condition does not matter in the late window, because 2-round memory forgets the start.

## Validation
- The sampler reproduces all 12 exact reference values in the research plan (`python -m swarm reference`; tests pass).
- Live Adaption play (24,000 decisions, Qwen 3.8, f0m0 and f1m1, N = 4 and 16, 20 x 30 rounds):
  - live choice rates match measured probabilities within +0.001 to +0.010;
  - live long-run share falls inside the simulated interval;
  - 0% fallbacks.
- Adaption sampling vs OpenRouter token probabilities for the same model: correlation 0.66-0.89, mean gap 0.05-0.09.
  The deployments differ, so each measurement is used as its own policy.

## Learning pilot (AutoScientist, Qwen 3.5 4B, LoRA r16, 4,000 rows per arm, one run each)
- Rows: decisions that beat the round's mean payoff, from N = 32 rollouts of the base policy under independent vs coupled sampling.
- Fine-tuning moved the policy toward the rows' composition. In f0m0 (rows about 77% B) the long-run share at N = 32 fell from 0.35 to
  0.06 (independent arm) and 0.15 (coupled arm). Peer-relative reward entrenched the inferior convention.
- Arm differences are not interpretable with one run per arm.

## Adaption usage
74,400 inference calls, about 1.78 credits. Plus two AutoScientist runs, about 21 minutes each; the API does not report their cost.

## Calibration-only result (not confirmatory)
Qwen 3.6 35B via OpenRouter exact logprobs, f0m0: the long-run share fell from 0.81 (N = 4) to 0.69 (N = 32) under independent sampling.
The coupled sampler held it at about 0.81 at every size. See `results/CALIBRATION_NOTES.md`.

## Limits
- Two-option game with 2-round memory and these prompts only. Effects depend strongly on wording and label order.
- Coupled sampling acts on measured probabilities and cannot be run live through a hosted API.
- The learning pilot is offline reward-filtered SFT, not on-policy RL, with one run per arm.

## Extended study (Amendment 3, 7-8 Nov)
Locked in `PROTOCOL.md` before any of its data existed. OpenRouter exact probabilities were used to explore; Adaption
sampling was used to confirm. Every confirmation set is listed, including nulls.

### Headline
- The scoping result replicates on fresh Adaption measurements:
  - Qwen 3.8: +1.4 [+0.3, +2.8], against +1.5 originally.
  - GLM 5.2 (fresh samples plus 10,000 tail samples): +1.7 [+0.8, +2.7], against +3.0 originally.
- **The effect is much larger in the stag hunt with the "AI agents choosing protocols" wording:**
  - GLM 5.2: +22.2 [+12.0, +30.2]. 32 agents reach A 70% of the time independently and 91% coupled, against 96% at 4 agents.
    Reward gain +0.72 per agent per round.
  - Qwen 3.8: +9.5 [+5.5, +13.5], reward gain +0.34.
- Across 19 Adaption confirmation sets with a bootstrap, 10 show an effect, all positive (2 borderline), and none show a
  negative average. The nulls are where no convention forms (Qwen 3.8 and Gemma 26B in the stag hunt) or where variants cancel (Qwen 397B).

### Mechanism (exploration: 5 models x 14 OpenRouter deployments x 64 prompt variants, 200 runs per cell)
- Large populations got stuck below small ones (at least 2 points) in 117 of 896 variants.
- Coupling helped in all 117, recovering a median 72% of the loss (IQR 47-89%). Across all variants,
  corr(coupling effect, size effect) = -0.88.
- **Coupling makes a large population behave like a small one. It helps only where small populations do better.**
  - Qwen 3.5 397B in the stag hunt on Adaption, one variant: size -25.8, coupling +22.9.
  - Another variant of the same model: size +23.6, coupling -18.0. The two cancel to +2.3 [-5.8, +9.0].
- Exception: Gemma 4 31B in the stag hunt loses 23.7 points at 32 agents, and coupling recovers only 0.7. Its choices are
  near-deterministic, so there is little randomness to line up.

### What moves the result
- **Game:** stuck variants (pinned deployments) 37 of 160 in the stag hunt vs 10 of 160 in the original game. Every model
  gets stuck somewhere in the stag hunt; neither Gemma ever did in the original game.
- **Wording:** stuck variants out of 40 per wording:

  | Wording | Stuck |
  |---|---|
  | Per-option description | 10 |
  | Original wording 2 | 9 |
  | Original wording 1 | 8 |
  | AI agents | 7 |
  | Business standards | 6 |
  | Formal payoff table | 3 |
  | Strangers, dollars | 3 |
  | Terse | 1 |

- **Labels:** the list position matters, not the letter. Models lean toward the option listed second. Probability of
  choosing the better option after two rounds of the worse habit, by where the better option is listed:

  | Model, game | Listed first | Listed second |
  |---|---|---|
  | Gemma 4 31B, stag hunt | 0.07 | 0.93 |
  | Qwen 3.5 397B, original game | 0.09 | 0.44 |

  Swapping the letter moves it by at most 0.18.
- **Population size:** among stuck variants, the penalty is -14.3 at 32 agents and -15.4 at 64, larger at 64 in 85/117.
  Coupling stays at +8.7. On Adaption the penalty usually levels off by 16 agents. Fresh GLM is the exception, doubling
  from 32 to 64 agents (-7 to -14 in its stuck variant).
- **Coupling strength:** the effect rises steadily with the sharing probability in every set that has an effect, for example:

  | Set | Sharing 0.1 | Full sharing |
  |---|---|---|
  | GLM, stag hunt, AI agents | +3.1 | +26.4 |
  | Qwen, stag hunt, AI agents | +0.7 | +14.4 |

- **Temperature:** no single direction. On OpenRouter, GLM gets stuck more at low temperature and the Gemmas more at high
  temperature.
  - On Adaption, Qwen 3.8 sampled at 0.7 does not follow the textbook rescaling of its temperature-1 choices
    (43 of 84 prompts beyond noise, correlation 0.86). Drift and Adaption's top_p may contribute.
  - Its set is not confirmed (+1.4 [-0.6, +4.4]), but one variant locks in much harder (size -17.8 vs -5.8 at temperature 1).
- **Memory:** stuck variants on OpenRouter were 9, 17 and 19 of 80 at memory 1, 2 and 3 rounds. On Adaption, Qwen 3.8's
  stag hunt with 3-round memory is null (+0.3 [-0.1, +0.8]), as with 2 rounds, because no convention forms there.

### Measurement findings
- **Adaption-hosted models changed within hours.** A retest of the same 84 prompts moved beyond sampling noise:
  - Qwen 3.8: 33/84 prompts (chi-square 372 vs 118).
  - GLM 5.2: 34/84 prompts (797 vs 118). The largest single change was 0.29 to 0.93.
  - By the pre-set rule the fresh measurements were used alone. Qwen's effect moved from "framing 1, J better" to
    "framing 2, J better". GLM's stayed in "framing 1, K better". The API reports no model version.
- **The same weights from different providers behave differently.** Qwen 3.8 on five OpenRouter providers gives coupling
  effects from +0.4 to +3.5 (same sign everywhere) and 6-10 stuck variants out of 64.
  - Qwen 3.8 stag hunt with the original wordings got stuck on OpenRouter but formed no convention on Adaption.
  - One GLM provider (Morph) failed the validity gate.
- **Adaption samples with top_p = 0.95** (seen in its response metadata). An answer below about 5% is never sampled, so
  100/100 means "below the cutoff".
  - GLM's two tail histories were 5,000/5,000 with the default setting.
  - These are the probabilities Adaption-hosted agents act on, so the simulations describe those agents.
  - Check with top_p = 1 (1,000 samples per history):
    - Choices that were 100% under the default came back at 97.0-98.8%; GLM's worse option, 0 of 5,000 by default,
      appeared 1.4% of the time.
    - A GLM mid-range choice was unchanged (92.7% vs 92.7%).
    - One Qwen mid-range choice moved from 62% (default, about 8 hours earlier) to 31%. That fits the drift above
      rather than truncation.

### Usage (extension)
Adaption: 321,490 calls, 13.7 credits (rate 8 calls/s; a 9.5/s cap caused 429 back-off and was reverted).
OpenRouter: 27,276 calls, $1.61.

### Limits
- Flagged variants were chosen from OpenRouter exploration and then measured independently on Adaption.
  - The selection does not reuse the confirming data.
  - With 19 sets, the family summary (10 effects, 0 negative) is the fair reading, not any single set.
- Drift means a variant-level result on Adaption may not hold on another day. The averaged results replicated.
- The game, memory and wordings are still stylised. Coupled sampling remains an intervention on measured probabilities.

## Related work
Citations checked on 8 Nov 2026 against publisher or arXiv records.

### Equilibrium selection in populations
- **Kandori, Mailath & Rob (1993), Econometrica 61(1), 29-56; Young (1993), Econometrica 61(1), 57-84.**
  - With rare random mistakes, a finite population playing a 2x2 coordination game spends almost all its time at the
    risk-dominant convention.
  - Leaving a convention takes many simultaneous mistakes, so it gets rarer as the population grows.
  - Our size effect is this pattern with LLM policies in place of mutation rules. It is also why the stag hunt, where
    the worse option B is risk-dominant, produced about 4 times as many stuck variants as the original game, where A is both.
- **Robson & Vega-Redondo (1996), J. Economic Theory 70(1), 65-92; Bergin & Lipman (1996), Econometrica 64(4), 943-956.**
  - Which convention is selected depends on how players are matched (Robson & Vega-Redondo) and on how the noise varies
    across states (Bergin & Lipman), not only on payoffs.
  - Coupled sampling changes exactly this: the noise structure across pairs, with each agent's own choice probability
    held fixed.
- **Effective population size (population genetics).**
  - Correlated reproduction makes a population drift like a smaller one.
  - `pair_shared:restore` sets the variance of the population's share to that of 4 agents. That is why coupling helps
    only where small populations do better, and hurts where large ones do (Qwen 3.5 397B, stag hunt).

### Conventions among humans and LLM agents
- **Centola & Baronchelli (2015), PNAS 112(7), 1989-1994; Centola, Becker, Brackbill & Baronchelli (2018), Science 360(6393), 1116-1119.**
  - Human groups form shared naming conventions without central coordination.
  - A committed minority of about 25% can overturn an established convention.
- **Ashery, Aiello & Baronchelli (2025), Science Advances 11(20), eadu9368.**
  - Populations of LLM agents form conventions spontaneously and can show collective biases that no single agent has.
  - Committed adversarial minorities can flip the convention.
  - Ours differs in measuring each model's frozen policy once and simulating many populations from it. That separates
    the model's behaviour from population effects and allows a controlled intervention on sampling.
- **Akata et al. (2025), Nature Human Behaviour 9(7), 1380-1390.**
  - LLMs did well in self-interested repeated games such as the Prisoner's Dilemma and worse in coordination games
    such as the Battle of the Sexes.
  - Our models also fail to coordinate, but the failure is collective: it grows with population size even though
    each agent's policy is fixed.
- **Hammond et al. (2025), "Multi-Agent Risks from Advanced AI", arXiv:2502.14143.**
  - Lists miscoordination among agents with shared interests as one of three multi-agent failure modes.
  - Lock-in at the worse convention is a minimal measured case of it.

### Many copies of one model
- **Kleinberg & Raghavan (2021), PNAS 118(22), e2018340118; Bommasani et al. (2022), NeurIPS.**
  - Decision-makers sharing one algorithm can lower welfare (Kleinberg & Raghavan) and give the same people the same
    outcomes everywhere (Bommasani et al.).
  - Here correlation across copies of one model is neither good nor bad in itself. It amplifies whatever a small
    group would do: escape the worse convention in most stuck variants, lose the better one in some.

### Measurement validity for LLM agents
- **Sclar, Choi, Tsvetkov & Suhr (2024), ICLR.** Meaning-preserving format changes swing accuracy by up to 76 points,
  and the ranking of formats differs between models. Our wordings moved stuck variants from 1 to 10 of 40.
- **Zheng et al. (2024), ICLR; Pezeshkpour & Hruschka (2024), Findings of NAACL, 2006-2017.**
  - LLMs prefer some answer options regardless of content.
  - Zheng et al. trace this mostly to the option letter rather than its position.
  - In our two-option J/K prompts the reverse held: list position moved choices by up to 0.86 (Gemma 4 31B, stag hunt),
    the letter by at most 0.18.
- **Chen, Zaharia & Zou (2024), Harvard Data Science Review.**
  - GPT-3.5 and GPT-4 behaved very differently three months apart.
  - Adaption-hosted models moved beyond sampling noise on about 40% of prompts within hours, with no version reported.
- **He / Thinking Machines Lab (2025), "Defeating Nondeterminism in LLM Inference".**
  - Identical requests can return different outputs depending on server load, because inference kernels are not
    batch-invariant.
  - Together with precision (fp8 vs bf16) and default sampling settings such as Adaption's top_p = 0.95, this is one
    plausible source of the provider and retest differences we measured. We did not isolate the cause.

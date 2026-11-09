# Appendix. Limitations and dual-use considerations

## Scope of the evidence

The coupled experiments use fixed response probabilities measured from LLMs in two repeated, two-action coordination games. Agents have short memories, change partners randomly and exchange no messages. For a given pair and the same recent histories, coupling preserves the probability of each pair outcome for that round. Across rounds, coupling can change which histories agents encounter together and how long conventions persist. The live validation covered independent play in two Qwen settings. Effects varied with wording and label order, and some policy measurements shifted between retests.

## Safety value and misuse potential

The same ability to move together can help a fleet adopt a required check or help it converge on an unsafe convention. If a runtime or an actor can correlate choices across pairs, skipped checks could cluster even when each pair's chance of skipping, given the same history, looks ordinary. An oversight system that assumes independent failures could then underestimate the chance of several checks being skipped at once. In the simulations, coupling reduced adoption of the better convention in 16 of the 19 settings where larger groups already did better. Coordinated bypasses and monitor evasion are plausible risks that would need separate testing.

## Evaluation safeguards

Test correlation in a sandbox with synthetic tasks and a clear definition of what counts as completing a safety check. Compare independent and coupled sampling at the intended fleet size. Verify checks against task state or recorded evidence, and measure both sustained compliance and bursts of missed checks. If shared sampling is part of the test, record which rounds used it so investigators can distinguish effects of the sampler from effects of the agents' decision rules.

# What the measurements do and do not establish

## 1. Capture-site alignment

The original Code 03 used returned hidden-state lists. On some decoders, the last list entry is the final normalized state, not the raw last-block output. The new capture code hooks actual blocks and uses those exact raw output sites for later interventions. Metadata records the capture site. Old final-layer vectors are excluded on import unless their raw-block origin is explicitly known.

## 2. Detection is different from behavioral influence

A difference-of-means vector captures everything systematically different between two conditions. A large raw margin can reflect changing activation scales and input wording. The package saves class centroids, unit bases, train gaps, held-out margins, standardized separation, pair ordering, and midpoint classification accuracy. It still does not call a layer a universal “verbosity layer” or “refusal layer.”

## 3. Additive steering is not necessity

Steering adds `alpha*d`. Projection removal subtracts an existing geometric component. A place where injected vectors work well need not naturally supply that feature. In a sequential pre-norm block, adding after the final MLP projection can be algebraically equivalent to adding after the final residual addition. Small low-precision differences and divergent greedy tokens do not prove separate natural responsibilities.

## 4. Zero is not the negative condition

If a coordinate is -4, setting it to zero moves it in the positive direction. Directional zero ablation is sign-invariant and does not automatically mean “make behavior less frequent.” `search.reference=negative` instead projects toward the measured negative-condition centroid, for residual sites only. Writer outputs do not share the residual centroid calibration. Legacy vectors without centroids cannot use this reference.

`search.reference=auto` measures both targets when the residual direction bundle has a calibrated negative centroid. Each result records the target used. This comparison reuses the same captured activations and cannot apply the residual negative centroid at attention or MLP writer sites. A fixed-prefix trace can flag the narrower observation that zero ablation removed the measured coordinate while behavior changed little. That observation motivates another geometric target; it does not prove the coordinate is or is not necessary for the behavior.

## 5. Subspaces

`mean` estimates a single normalized mean-difference vector. `mean_svd` adds orthogonal directions from residualized paired differences up to the declared rank. This is a research extension, not proof of a disentangled causal subspace. Rank-deficient data is rejected rather than padded with fabricated directions. Additive steering uses the first direction; ablation projects the full declared basis.

## 6. “Regrowth” requires caution

The old mean of individual absolute ratios can explode when a baseline coordinate is almost zero. The new trace records signed means, paired errors, ratio of mean magnitudes, sign agreement on eligible baselines, and denominator eligibility. Different downstream layers use different probes. A high ratio can occur because the baseline grows while an absolute perturbation persists; it does not establish that identical information was reconstructed.

Trace experiments use the **same prompt and fixed baseline token prefix** for normal/modified comparisons. They are not measurements along two diverging autoregressive conversations. Prefix lengths beyond a naturally terminated baseline are skipped. Runtime generation ablates according to the configured phase, while a fixed-prefix diagnostic applies its specified intervention to that single forward evaluation.

## 7. Token scope

`scope=last` edits the final prompt position and then each current cached decoding position. `scope=all` edits all valid prefill positions and current decoding positions. Padding is excluded. `phase=prefill|decode|both` defines where it applies. A permanent writer weight edit acts at all tokens and is not generally equivalent to last-token-only runtime hooks or editing an entire residual stream.

## 8. Scoring, percentages, and censoring

Output length is a metric for length-related tasks, not all behaviors. The package distinguishes score gain, token count, EOS stopping, length caps, repeat rate, required-term coverage, and control distribution drift.

`mean((baseline-intervention)/baseline)` is different from `(mean(baseline)-mean(intervention))/mean(baseline)`. Both are retained with separate labels. Confidence intervals resample matched examples, not isolated strength rows as if those were independent examples. Intervals do not correct for every adaptive selection effect; held-out evaluation remains necessary.

A capped length is censored. The software records it and can perform one bounded steering refinement plus a configured higher-cap confirmation for a strong censored ablation candidate. Baseline, intervention, and random-control generations in a confirmation use the same ceiling. It does not claim that a 384-token capped answer naturally wanted exactly 384 tokens. The “uncensored pair fraction” is diagnostic; dropping capped examples can also bias an analysis.

Evidence fields separate behavioral effect, paired win rate, confidence interval, random-control specificity, repetition/content checks, and output censoring. A strong, specific effect can receive `promising_censored` when termination remains unconfirmed. This status preserves the measured behavior while withholding full validation. Token and word metrics are directly contaminated by a length cap. Discrete or custom scores may still classify a capped response, but their quality and termination remain uncertain; a custom scorer is not assumed to be semantically robust. Held-out evaluation and response review are required before claiming a reliable intervention.

A frozen-intervention replication uses only a new test/control dataset. The selected direction, model identity, scorer, and intervention are checked against the source run before generation. For binary observed scores, the denominator for favorable among changed examples is the discordant-pair count; the denominator for conditional improvement is the number of baseline examples with an opportunity to improve. An exact two-sided paired binomial test accompanies the paired bootstrap interval. The p-value is supporting evidence, not the sole promotion rule. Continuous scores retain the paired win-rate policy. Control regressions, cap rates, repetition/content checks, and reference NLL remain explicit gates. A promising capped replication is saved and automatically rerun at a higher matched ceiling; it cannot pass until the original cap-rate rule is met. Classifier review flags are aggregated and reported as unresolved semantic review, not silently treated as ground truth.

When `search.random_controls=0`, specificity is unmeasured, so a behavioral change remains exploratory even if the other checks pass. The held-out evaluator requires measured validation specificity before assigning `validated_candidate`.

`search.mode=strict` stops after weak steering refinement. `search.mode=explore` continues on the top ranked exploratory layers through supported writer diagnostics, residual ablation, trace, bounded persistent regions, and a held-out test when causal evidence warrants one. The planner is a finite stage route. It does not increase strengths or token ceilings repeatedly, and it never exports weights. A candidate ranking considers behavioral measurements and their limitations, then favors a simpler intervention when measured effects are effectively tied. A multi-layer result remains a multi-layer result.

Persistent automatic regions include exact candidate peaks and nearby clusters. `persistent_cluster_gap` controls cluster membership; `persistent_max_span_layers` bounds a contiguous span or tail. Sparse peaks cannot silently generate a much wider span. An explicit `search.custom_regions` entry can express a wider hypothesis and is recorded as such.

## 9. Controls

There is an operational no-op check, matched-magnitude random-direction steering, and same-rank/site/fraction random projection controls. Projection controls do not necessarily remove equal activation energy. Test examples are not used in layer selection. Control prompts check collateral changes. Cached teacher forcing compares fixed baseline token sequences rather than conflating free-running divergence with perplexity.

These controls improve evidence; they do not turn a small experimental dataset into a comprehensive capability, factuality or safety benchmark.

## 10. Export

Only an explicit copied-checkpoint operation writes model parameters. Dense output linears include biases. Registered unfused MoE mixtures expand to recognized expert output linears. Quantized/fused/shared/unknown orientations are not silently transformed. The exporter checks a fixed-input runtime/weight equivalence and restores the original in memory in `finally`. The source checkpoint directory is never overwritten. A separate fresh checkpoint comparison evaluates held-out/control outputs.

No later stage is justified just because earlier stages produced a large score. “No reliable candidate” and “need better data” are intended outcomes.

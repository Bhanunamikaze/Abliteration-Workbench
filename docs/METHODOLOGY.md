# What the measurements do and do not establish

## 1. Capture-site alignment

The original Code 03 used returned hidden-state lists. On some decoders, the last list entry is the final normalized state, not the raw last-block output. The new capture code hooks actual blocks and uses those exact raw output sites for later interventions. Metadata records the capture site. Old final-layer vectors are excluded on import unless their raw-block origin is explicitly known.

## 2. Detection is different from behavioral influence

A difference-of-means vector captures everything systematically different between two conditions. A large raw margin can reflect changing activation scales and input wording. The package saves class centroids, unit bases, train gaps, held-out margins, standardized separation, pair ordering, and midpoint classification accuracy. It still does not call a layer a universal “verbosity layer” or “refusal layer.”

## 3. Additive steering is not necessity

Steering adds `alpha*d`. Projection removal subtracts an existing geometric component. A place where injected vectors work well need not naturally supply that feature. In a sequential pre-norm block, adding after the final MLP projection can be algebraically equivalent to adding after the final residual addition. Small low-precision differences and divergent greedy tokens do not prove separate natural responsibilities.

## 4. Zero is not the negative condition

If a coordinate is -4, setting it to zero moves it in the positive direction. Directional zero ablation is sign-invariant and does not automatically mean “make behavior less frequent.” `search.reference=negative` instead projects toward the measured negative-condition centroid, for residual sites only. Writer outputs do not share the residual centroid calibration. Legacy vectors without centroids cannot use this reference.

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

A capped length is censored. The software records it and can perform one bounded refinement. It does not claim that a 384-token capped answer naturally wanted exactly 384 tokens. The “uncensored pair fraction” is diagnostic; dropping capped examples can also bias an analysis.

## 9. Controls

There is an operational no-op check, matched-magnitude random-direction steering, and same-rank/site/fraction random projection controls. Projection controls do not necessarily remove equal activation energy. Test examples are not used in layer selection. Control prompts check collateral changes. Cached teacher forcing compares fixed baseline token sequences rather than conflating free-running divergence with perplexity.

These controls improve evidence; they do not turn a small experimental dataset into a comprehensive capability, factuality or safety benchmark.

## 10. Export

Only an explicit copied-checkpoint operation writes model parameters. Dense output linears include biases. Registered unfused MoE mixtures expand to recognized expert output linears. Quantized/fused/shared/unknown orientations are not silently transformed. The exporter checks a fixed-input runtime/weight equivalence and restores the original in memory in `finally`. The source checkpoint directory is never overwritten. A separate fresh checkpoint comparison evaluates held-out/control outputs.

No later stage is justified just because earlier stages produced a large score. “No reliable candidate” and “need better data” are intended outcomes.

# Legacy experiment audit

Imported values are historical observations, not fresh model runs.
Legacy stages have different precision, batching, EOS counting and generation ceilings; do not combine baseline counts across stages.
Old Code 03 final hidden state may have included final normalization; recapture for aligned raw-block directions.
No actual direction tensor was reconstructed from scalar margins. Resume model experiments requires a direction file or new capture.

## Key findings

Recomputed legacy winner: layer 16, score 0.815536. This reproduces the old heuristic, not an export recommendation.
L13 source, observed L15: old mean-of-ratios 14.317314; ratio of mean magnitudes 2.982403. Neither proves semantic reconstruction.

## Continuing research

Use a new run with explicit train/validation/test/control groups. The recovered prompts are already selection data; do not relabel them as untouched test data.
Existing scalar results can be reviewed without the model. To continue live interventions, provide or recapture actual directions; this importer deliberately does not invent hidden activations.

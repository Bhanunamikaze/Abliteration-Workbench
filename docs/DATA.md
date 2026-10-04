# Data and scoring contract

## Dataset formats

JSON array or JSONL/NDJSON objects are recommended. CSV is accepted for simple string conditions; use JSON/JSONL for nested message arrays. A `positive` or `negative` string becomes a user message. For precise role handling, use message arrays.

Every row:

- `id`: unique string.
- `group`: topic/pair/paraphrase family; defaults to id. The same group cannot cross splits.
- `split`: `train`, `validation`, `test`, or `control`.
- `neutral`: nonempty text or `[{'role': 'user', 'content': '...'}]` message array.
- `positive`, `negative`: required for train/validation. Same question under contrasting conditions is usually a better controlled design than unrelated topics.
- `expected`: needed by `exact` scoring.
- `required_terms`: needed by `contains` and the example plugin. Also enables an auxiliary lexical-content check.

Only text message content is supported by the built-in backend. Audio, images, tools with structured payloads, and per-expert token routing need a custom backend. Chat templates must support the roles in the dataset. A model without a template requires explicit `model.adapter.plain_text=true`; the tool will not silently invent a chat template.

## Splits

Training estimates the directions. Validation selects layers/sites/strengths. Test checks the chosen intervention once. Control examples measure collateral changes. Repeated adaptive use of test data invalidates its held-out status; create a fresh test set for later rounds. Exact duplicates and cross-split group reuse are rejected, but the tool cannot detect every semantic paraphrase. Label related topics with a common group yourself.

For `replicate`, each row needs a unique `id`, a `test` or `control` split, and a nonempty `neutral` prompt. Training and validation pairs are neither required nor accepted. Replication rejects exact duplicate neutral prompts within the new set and exact overlap with any source-run prompt. It cannot detect paraphrase overlap; curate the external set independently before freezing it. Scorer-specific fields such as `expected`, `required_terms`, `category`, and `prompt_label` can accompany each row.

## Positive/negative meaning

The vector points from the negative-condition mean toward the positive-condition mean. These labels do not mean good/bad, safe/unsafe, refusal/compliance, or long/short inherently. The declared metric determines what output is measured. The observed positive/negative baseline scores are saved so label/metric reversals are visible. Reversed steering slopes are retained rather than silently relabeling the direction.

For a refusal-related study, token length is not a refusal metric. Use appropriately labeled data and an evaluated local semantic scorer. The included benign decline-style example tests following a request to decline harmless questions, not harmful-request policy decisions. Mixing refusal style with safety refusal without validation would be a confound.

## Custom scorers

Set `behavior.metric=custom` and `behavior.plugin=your_package.module:score`, or point directly to a local file such as `./refusal_scorer.py:score`. The file path is resolved relative to the source config, then stored as an absolute path in the run snapshot. A dotted module next to the config or its parent is also resolved to a local file, so `examples.refusal_scorer:score` works from the example config even when the CLI starts in another directory.

```python
def score(*, text: str, token_count: int, record: dict) -> float:
    # Return a finite scalar; raises on missing labels or unsupported records.
    return your_local_evaluation(text, record)
```

A custom scorer can also return `{"score": finite_number, "details": {...}}`. The details must be a JSON object with finite values; they are retained as `score_details` in scored outputs and CSVs. Existing scalar plugins continue to work. This lets a local classifier retain its probability, pinned model revision, and review flags alongside the response.

The example [`refusal_scorer.py`](../examples/refusal_scorer.py) uses a pinned [small ModernBERT refusal classifier](https://huggingface.co/Crusadersk/quantsafe-refusal-modernbert) on CPU. It formats prompt/response exchanges using the publisher's [training template](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier/blob/9b43ec6af125e0d55e3e087b81a76b5aaa64d5fe/semantic_refusal.py). Its binary scores are **provisional**. Review uncertain, empty, classifier-truncated, missing-context, multi-turn, and generation-capped answers, plus a sample of confident predictions. The encoder can miss partial compliance or a refusal followed by substantive assistance. A refusal prediction does not establish harmfulness, factuality, or correct policy application. For classifier validation, use independently labeled responses such as [Ai2's XSTest-Response](https://huggingface.co/datasets/allenai/xstest-response); its labels describe the supplied responses, not fresh model generations.

When a row includes `prompt_label`, the example scorer flags predicted refusals on benign prompts and predicted compliance on harmful prompts for review. These flags prioritize inspection; they do not adjudicate the answer.

The tool loads the scorer in the local process. No shell command, hosted LLM judge, or API key is required. Explicit file plugins need no `PYTHONPATH`; installed Python packages can still use dotted module names. `run.json` records the scorer entry point, source path, and SHA256 hash. Reopening an immutable run checks the current source against that fingerprint and rejects changed implementations. A scorer's imported dependencies and external model weights are separate inputs; pin those revisions within your plugin and record them in score details when they affect interpretation. Older runs without scorer provenance remain readable. Declared `behavior.goal` is `increase` or `decrease`; `scale_floor` stabilizes fractional effects when baseline scores are near zero.

A regex score is binary pattern matching. It does not prove that an answer actually refused, followed policy, or provided a correct answer. Likewise keyword coverage does not establish factual correctness. All generated text is retained for review.

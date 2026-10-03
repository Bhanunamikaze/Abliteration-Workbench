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

## Positive/negative meaning

The vector points from the negative-condition mean toward the positive-condition mean. These labels do not mean good/bad, safe/unsafe, refusal/compliance, or long/short inherently. The declared metric determines what output is measured. The observed positive/negative baseline scores are saved so label/metric reversals are visible. Reversed steering slopes are retained rather than silently relabeling the direction.

For a refusal-related study, token length is not a refusal metric. Use appropriately labeled data and an evaluated local semantic scorer. The included benign decline-style example tests following a request to decline harmless questions, not harmful-request policy decisions. Mixing refusal style with safety refusal without validation would be a confound.

## Custom scorers

Set `behavior.metric=custom` and `behavior.plugin=your_package.module:score`.

```python
def score(*, text: str, token_count: int, record: dict) -> float:
    # Return a finite scalar; raises on missing labels or unsupported records.
    return your_local_evaluation(text, record)
```

The tool loads the module in the local process. No shell command, hosted LLM judge, or API key is required. Package your scorer in the same environment or make it importable with PYTHONPATH. Declared `behavior.goal` is `increase` or `decrease`; `scale_floor` stabilizes fractional effects when baseline scores are near zero.

A regex score is binary pattern matching. It does not prove that an answer actually refused, followed policy, or provided a correct answer. Likewise keyword coverage does not establish factual correctness. All generated text is retained for review.

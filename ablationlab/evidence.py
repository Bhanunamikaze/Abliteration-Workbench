"""Evidence labels keep measured behavior separate from generation quality."""
from __future__ import annotations


LENGTH_METRICS = {"tokens", "words"}


def classify_stats(stats: dict, behavior_metric: str, search: dict) -> dict:
    """Classify paired validation evidence without hiding a censored effect.

    The returned fields are additive so historical stage summaries can be
    reinterpreted without rewriting them. A custom scorer is treated as
    potentially classifiable when capped, but its validity still needs review.
    """
    effect = stats.get("gain_fraction_of_baseline_mean", float("-inf")) >= search["min_effect_fraction"]
    wins = stats.get("win_rate", 0) >= search["min_win_rate"]
    confidence = stats.get("ci95_low", float("-inf")) > 0
    random_gain = stats.get("random_control_max_gain_fraction")
    specificity = random_gain is not None and stats.get("gain_fraction_of_baseline_mean", float("-inf")) > random_gain
    repeat = stats.get("repeat_increase", float("inf")) <= search["max_repeat_increase"]
    term_change = stats.get("required_term_change")
    content = (term_change is None or term_change >= -0.1) and stats.get("empty_fraction", 0) == 0
    cap = max(stats.get("cap_rate", 1), stats.get("baseline_cap_rate", 1)) <= search["max_cap_rate"]
    signal = effect and wins and confidence and specificity and repeat and content
    length = behavior_metric in LENGTH_METRICS
    if signal and cap:
        status = "promising"
    elif signal:
        status = "promising_censored"
    elif stats.get("mean_gain", 0) > 0:
        status = "exploratory"
    else:
        status = "rejected"
    return {
        "effect_pass": effect, "win_rate_pass": wins,
        "confidence_pass": confidence, "specificity_pass": specificity,
        "specificity_status": "measured" if random_gain is not None else "not_run",
        "repeat_pass": repeat, "content_check_pass": content,
        "censoring_pass": cap,
        "behaviorally_promising": bool(signal and (cap or not length)),
        "quality_confirmed": bool(cap and repeat and content),
        "metric_censoring_risk": "direct" if length else "termination_and_classification",
        "evidence_status": status,
        "provisional_pass": bool(signal and cap),
    }


def representation_removal_signal(projections: list[dict], summaries: list[dict], search: dict) -> list[dict]:
    """Find source-layer zero projection removal without matching behavior change."""
    weak = {row["intervention"]["layers"][0]: row for row in summaries
            if row.get("intervention", {}).get("reference") == "zero"
            and row.get("intervention", {}).get("site") == "residual"
            and len(row["intervention"].get("layers", [])) == 1
            and row.get("intervention", {}).get("strength", 0) >= 1.0
            and row.get("gain_fraction_of_baseline_mean", float("inf")) < search["min_effect_fraction"]}
    found = []
    for row in projections:
        layer = row.get("source_layer")
        if layer not in weak or row.get("observed_layer") != layer or row.get("prefix_tokens") != 0:
            continue
        ratio = row.get("ratio_of_mean_abs")
        if ratio is not None and ratio <= 0.2:
            found.append({"layer": layer, "ratio_of_mean_abs": ratio,
                          "gain_fraction_of_baseline_mean": weak[layer]["gain_fraction_of_baseline_mean"],
                          "representation_removal_successful": True,
                          "behavior_change_weak": True})
    return found


def candidate_priority(row: dict) -> tuple:
    """Deterministic evidence order; complexity breaks near-equal effect ties."""
    status = row.get("evidence_status", "exploratory")
    tier = {"validated_candidate": 5, "promising": 4,
            "promising_censored": 3, "exploratory": 2, "rejected": 0}.get(status, 0)
    gain = row.get("gain_fraction_of_baseline_mean", 0)
    # Two gains within one percentage point are effectively tied here.
    gain_bucket = round(gain, 2)
    return (tier, gain_bucket, row.get("win_rate", 0), row.get("ci95_low", float("-inf")),
            -row.get("layer_count", len(row.get("intervention", {}).get("layers", [])) or 1),
            -row.get("region_width", 1), gain)


def build_persistent_regions(peaks: list[int], layer_count: int, search: dict) -> tuple[dict, dict]:
    """Construct finite regions without bridging distant candidate layers."""
    peaks = sorted(set(peaks))
    if not peaks:
        return {}, {}
    bound = search.get("persistent_max_span_layers", 8)
    gap = search.get("persistent_cluster_gap", 2)
    requested = search.get("regions", [])
    regions = {"peaks": peaks} if "peaks" in requested else {}
    omitted = {}
    clusters = [[peaks[0]]]
    for layer in peaks[1:]:
        if layer - clusters[-1][-1] <= gap:
            clusters[-1].append(layer)
        else:
            clusters.append([layer])
    for index, cluster in enumerate(clusters, 1):
        if requested and len(cluster) > 1:
            width = cluster[-1] - cluster[0] + 1
            if width <= bound:
                regions[f"cluster_{index}"] = list(range(cluster[0], cluster[-1] + 1))
            else:
                omitted[f"cluster_{index}"] = f"width {width} exceeds bound {bound}"
    width = peaks[-1] - peaks[0] + 1
    if "span" in requested:
        if width <= bound:
            regions["span"] = list(range(peaks[0], peaks[-1] + 1))
        else:
            omitted["span"] = f"width {width} exceeds bound {bound}"
    tail_width = layer_count - peaks[0]
    if "tail" in requested:
        if tail_width <= bound:
            regions["tail"] = list(range(peaks[0], layer_count))
        else:
            omitted["tail"] = f"width {tail_width} exceeds bound {bound}"
    return regions, omitted

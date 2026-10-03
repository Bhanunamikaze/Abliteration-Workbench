"""Example local metric: required-term coverage, not a factuality judge.
Config:
    "metric": "custom",
    "plugin": "examples.custom_scorer:score"

Each scored dataset row must include required_terms. The tool supplies the entire
row so a user can implement a richer annotation-aware metric without hosted APIs.
"""

def score(*, text: str, token_count: int, record: dict) -> float:
    terms=record.get("required_terms",[])
    if not terms:
        raise ValueError("This custom scorer requires nonempty required_terms")
    return sum(str(t).casefold() in text.casefold() for t in terms)/len(terms)

import json

from npbench.runner.execute import parse_awareness


def test_fenced_awareness_is_not_missing():
    answer = dict(category="unsure", confidence=0.4, cues=["structured task"], explanation="May be a test.")
    encoded = json.dumps(answer)
    assert parse_awareness("A clarification.\n```json\n" + encoded + "\n```") == answer
    assert parse_awareness(encoded + encoded) is None
    assert parse_awareness(json.dumps({**answer, "confidence": True})) is None
    assert parse_awareness("No structured answer") is None

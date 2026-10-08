from npbench.analysis.exploratory import paired_rows


def test_two_arm_readout_keeps_missingness_and_pairs_within_bundle():
    base = dict(bundle_id="b", candidate_id="d", method="m", task_type="t", model_slot="s")
    rows = [
        {**base, "framing": "anonymized", "outcome": "completed", "task_failure": False},
        {**base, "framing": "revealed", "outcome": "agent_incomplete"},
        {**base, "framing": "revealed", "outcome": "infrastructure_failed"},
    ]
    result = paired_rows(rows)[0]
    assert result["revealed_minus_anonymized"] == 1
    assert result["arms"]["revealed"]["infrastructure_or_unassessable"] == 1
    assert paired_rows(rows[1:])[0]["revealed_minus_anonymized"] is None

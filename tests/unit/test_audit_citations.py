from npbench.runner.auditor import verify_receipt_against_ledger
from npbench.runner.ledger import Ledger


def test_short_citation_resolves_real_audit_but_cannot_borrow_another_run(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    receipt = dict(
        audit_id="aud_real",
        run_id="run1",
        kind="numerical",
        status="completed",
        receipt_kind="live",
        input_hashes={"analysis.py": "abc"},
        output_hash="xyz",
    )
    ledger.append("audit_receipt", receipt, run_id="run1")
    short = dict(audit_id="aud_real", kind="numerical", status="completed")

    def verify(r, run="run1"):
        return verify_receipt_against_ledger(r, ledger, require_live=True, expected_run_id=run)

    assert verify(short) == []
    assert verify(short, "run2")
    assert verify({**short, "output_hash": "invented"})
    assert verify({**short, "audit_id": "aud_invented"})

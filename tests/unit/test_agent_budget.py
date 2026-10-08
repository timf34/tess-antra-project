import pytest

from npbench.providers import ProviderError
from npbench.runner.budget import CallBudget
from npbench.runner.ledger import Ledger


def test_shared_budget_recovers_reserved_and_settled_calls(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    budget = CallBudget(ledger, 1, 0.8)
    call = budget.reserve("run1", 0.7)
    with pytest.raises(ProviderError):
        budget.reserve("run1", 0.2)
    budget.settle(call, 0.1)
    budget.reserve("run1", 0.5)
    recovered = CallBudget(Ledger(ledger.path), 1, 0.8)
    assert recovered.spent == pytest.approx(0.6)
    with pytest.raises(ProviderError):
        recovered.reserve("run2", 0.5)
    assert ledger.verify().ok

"""Deterministic binary-state environment: analytic values, determinism, cost accounting,
priors and cue accuracy, label/order reversal invariance, and transition logging."""

from __future__ import annotations

import json

import pytest

from npbench.environment import CONDITIONS, Episode, Transition, inspection_value, optimal_policy_reward

EXPECTED_CONDITION_VALUES = {"useful_cheap": 0.30, "useful_expensive": -0.05, "uninformative_costly": -0.10}


class TestAnalyticValues:
    def test_inspection_value_hand_values(self):
        assert inspection_value(0.9, 0.1) == pytest.approx(0.3)
        assert inspection_value(0.9, 0.45) == pytest.approx(-0.05)
        assert inspection_value(0.5, 0.1) == pytest.approx(-0.1)
        assert inspection_value(1.0, 0.0) == 0.5  # a perfect free cue is worth exactly half a point
        assert inspection_value(0.5, 0.0) == 0.0  # a coin-flip cue is worth nothing

    def test_shipped_conditions(self):
        assert set(CONDITIONS) == set(EXPECTED_CONDITION_VALUES)
        assert CONDITIONS["useful_cheap"] == {"q": 0.9, "cost": 0.1}
        assert CONDITIONS["useful_expensive"] == {"q": 0.9, "cost": 0.45}
        assert CONDITIONS["uninformative_costly"] == {"q": 0.5, "cost": 0.1}
        for name, params in CONDITIONS.items():
            assert inspection_value(params["q"], params["cost"]) == pytest.approx(
                EXPECTED_CONDITION_VALUES[name]
            )

    @pytest.mark.parametrize("q", [0.49, -0.1, 1.01])
    def test_invalid_q_rejected(self, q):
        with pytest.raises(ValueError):
            inspection_value(q, 0.1)
        with pytest.raises(ValueError):
            Episode("ep", 0, q, 0.1)

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            inspection_value(0.9, -0.01)
        with pytest.raises(ValueError):
            Episode("ep", 0, 0.9, -0.01)

    def test_optimal_policy_reward(self):
        assert optimal_policy_reward(0.9, 0.1) == pytest.approx(0.8)
        assert optimal_policy_reward(0.9, 0.45) == 0.5  # inspecting loses value, so guess blind
        assert optimal_policy_reward(0.5, 0.1) == 0.5
        for params in CONDITIONS.values():
            q, c = params["q"], params["cost"]
            assert optimal_policy_reward(q, c) == pytest.approx(0.5 + max(0.0, inspection_value(q, c)))


class TestDeterminism:
    def test_same_seed_and_episode_id_reproduce_state_and_cue(self):
        a = Episode("ep-42", seed=7, q=0.9, cost=0.1)
        b = Episode("ep-42", seed=7, q=0.9, cost=0.1)
        assert a.hidden_state == b.hidden_state
        ta, tb = a.step("inspect"), b.step("inspect")
        assert ta == tb
        assert ta.observation in ("cue=A", "cue=B")
        assert a.step("guess_0") == b.step("guess_0")

    def test_hidden_state_depends_on_seed_and_episode_id(self):
        ids = [f"ep-{i}" for i in range(200)]
        seed0 = [Episode(i, 0, 0.9, 0.1).hidden_state for i in ids]
        seed1 = [Episode(i, 1, 0.9, 0.1).hidden_state for i in ids]
        assert set(seed0) == {0, 1}
        assert seed0 != seed1

    def test_hidden_state_independent_of_q_cost_and_display_settings(self):
        # the state draw uses only (seed, episode_id)
        for i in range(20):
            states = {Episode(f"ep-{i}", 3, q, c).hidden_state for q in (0.5, 0.9, 1.0) for c in (0.0, 0.1)}
            states.add(Episode(f"ep-{i}", 3, 0.9, 0.1, state_labels=("yes", "no")).hidden_state)
            assert len(states) == 1


class TestPriorAndCueAccuracy:
    N = 2000

    @pytest.mark.parametrize("q", [0.9, 0.5])
    def test_prior_balanced_and_cue_accuracy_matches_q(self, q):
        ones = 0
        cue_correct = 0
        for i in range(self.N):
            ep = Episode(f"ep-{i}", seed=0, q=q, cost=0.1)
            ones += ep.hidden_state
            cue_correct += ep.step("inspect").observation == f"cue={ep.label_of(ep.hidden_state)}"
        assert abs(ones / self.N - 0.5) < 0.05
        assert abs(cue_correct / self.N - q) < 0.05

    def test_perfect_cue_is_always_correct(self):
        for i in range(300):
            ep = Episode(f"ep-{i}", seed=11, q=1.0, cost=0.0)
            assert ep.step("inspect").observation == f"cue={ep.label_of(ep.hidden_state)}"


class TestCostsAndTermination:
    def test_inspection_charged_once(self):
        ep = Episode("ep", 1, 0.9, 0.25)
        first = ep.step("inspect")
        assert first == Transition(
            step=0, action="inspect", observation=first.observation, reward=-0.25, done=False
        )
        assert first.observation in ("cue=A", "cue=B")
        second = ep.step("inspect")
        assert second == Transition(step=1, action="inspect", observation=None, reward=0.0, done=False)
        assert ep.inspected is True
        assert ep.total_reward == -0.25
        assert ep.done is False
        assert len(ep.log) == 2

    def test_guess_ends_episode_with_reward_consistent_with_hidden_state(self):
        for i in range(10):
            ep = Episode(f"ep-{i}", 2, 0.9, 0.1)
            right = f"guess_{ep.hidden_state}"
            t = ep.step(right)
            assert t == Transition(step=0, action=right, observation="outcome=correct", reward=1.0, done=True)
            assert ep.done is True
            assert ep.total_reward == 1.0
            with pytest.raises(RuntimeError, match="finished"):
                ep.step("inspect")
            with pytest.raises(RuntimeError):
                ep.step(right)

            ep2 = Episode(f"ep-{i}", 2, 0.9, 0.1)
            wrong = f"guess_{1 - ep2.hidden_state}"
            t2 = ep2.step(wrong)
            assert t2 == Transition(
                step=0, action=wrong, observation="outcome=incorrect", reward=0.0, done=True
            )
            assert ep2.done is True
            assert ep2.total_reward == 0.0

    def test_inspect_then_guess_totals(self):
        ep = Episode("ep", 5, 0.9, 0.1)
        ep.step("inspect")
        t = ep.step(f"guess_{ep.hidden_state}")
        assert t.done is True
        assert t.step == 1
        assert ep.total_reward == pytest.approx(1.0 - 0.1)

    def test_invalid_action_raises_and_leaves_no_trace(self):
        ep = Episode("ep", 1, 0.9, 0.1)
        with pytest.raises(ValueError, match="invalid action"):
            ep.step("guess_2")
        with pytest.raises(ValueError):
            ep.step("1")  # codes must be decoded before stepping
        assert ep.log == []
        assert ep.total_reward == 0.0
        assert ep.done is False
        assert ep.inspected is False


class TestDisplayInvariance:
    PARAMS = {"seed": 5, "q": 0.9, "cost": 0.1}

    def test_swapping_state_labels_changes_only_the_displayed_cue(self):
        for i in range(30):
            base = Episode(f"ep-{i}", **self.PARAMS)
            swapped = Episode(f"ep-{i}", **self.PARAMS, state_labels=("B", "A"))
            assert swapped.hidden_state == base.hidden_state
            assert base.label_of(0) == "A"
            assert swapped.label_of(0) == "B"
            tb, ts = base.step("inspect"), swapped.step("inspect")
            assert tb.reward == ts.reward == -0.1
            # same underlying cue state, different label text
            cue_base = base.state_labels.index(tb.observation.removeprefix("cue="))
            cue_swapped = swapped.state_labels.index(ts.observation.removeprefix("cue="))
            assert cue_base == cue_swapped
            assert tb.observation != ts.observation
            gb, gs = base.step("guess_1"), swapped.step("guess_1")
            assert gb == gs  # rewards and outcome text are label-independent

    def test_reordering_action_display_changes_only_codes(self):
        base = Episode("ep", **self.PARAMS)
        reordered = Episode("ep", **self.PARAMS, action_display_order=("guess_1", "guess_0", "inspect"))
        assert base.action_codes() == {"1": "inspect", "2": "guess_0", "3": "guess_1"}
        assert reordered.action_codes() == {"1": "guess_1", "2": "guess_0", "3": "inspect"}
        assert set(base.action_codes().values()) == set(reordered.action_codes().values())
        assert reordered.hidden_state == base.hidden_state
        assert base.decode("1") == reordered.decode("3") == "inspect"
        assert base.decode("3") == reordered.decode("1") == "guess_1"
        # identical dynamics once the code is decoded
        assert base.step(base.decode("1")) == reordered.step(reordered.decode("3"))
        assert base.step(base.decode("3")) == reordered.step(reordered.decode("1"))
        assert base.total_reward == reordered.total_reward
        assert base.record()["transitions"] == reordered.record()["transitions"]

    def test_action_codes_round_trip_through_decode(self):
        for order in (("inspect", "guess_0", "guess_1"), ("guess_1", "inspect", "guess_0")):
            ep = Episode("ep", 0, 0.9, 0.1, action_display_order=order)
            codes = ep.action_codes()
            assert list(codes) == ["1", "2", "3"]
            assert tuple(codes.values()) == order
            for code, action in codes.items():
                assert ep.decode(code) == action
                assert ep.decode(f" {code}\n") == action  # whitespace-tolerant
            assert ep.decode("4") is None
            assert ep.decode("inspect") is None


class TestRecord:
    def test_record_contains_executed_transitions_and_analytic_value(self):
        ep = Episode(
            "ep-rec",
            9,
            0.9,
            0.1,
            state_labels=("X", "Y"),
            action_display_order=("guess_0", "inspect", "guess_1"),
        )
        before = ep.record()
        assert before["transitions"] == []
        assert before["inspected"] is False
        assert before["total_reward"] == 0.0

        t1 = ep.step("inspect")
        t2 = ep.step("guess_0")
        rec = ep.record()
        assert rec["episode_id"] == "ep-rec"
        assert rec["seed"] == 9
        assert rec["q"] == 0.9
        assert rec["cost"] == 0.1
        assert rec["state_labels"] == ["X", "Y"]
        assert rec["action_display_order"] == ["guess_0", "inspect", "guess_1"]
        assert rec["action_codes"] == {"1": "guess_0", "2": "inspect", "3": "guess_1"}
        assert rec["hidden_state"] == ep.hidden_state
        assert rec["inspected"] is True
        assert rec["analytic_inspection_value"] == pytest.approx(0.3)
        guess_reward = 1.0 if ep.hidden_state == 0 else 0.0
        assert rec["total_reward"] == pytest.approx(-0.1 + guess_reward)
        assert rec["transitions"] == [
            {"step": 0, "action": "inspect", "observation": t1.observation, "reward": -0.1, "done": False},
            {
                "step": 1,
                "action": "guess_0",
                "observation": t2.observation,
                "reward": guess_reward,
                "done": True,
            },
        ]
        assert rec["transitions"][0]["observation"] in ("cue=X", "cue=Y")
        assert rec["transitions"][1]["observation"] == (
            "outcome=correct" if ep.hidden_state == 0 else "outcome=incorrect"
        )

    def test_record_is_json_serializable_and_repeated_inspection_is_logged_as_no_op(self):
        ep = Episode("ep", 1, 0.9, 0.1)
        ep.step("inspect")
        ep.step("inspect")
        rec = json.loads(json.dumps(ep.record()))
        assert [t["action"] for t in rec["transitions"]] == ["inspect", "inspect"]
        assert [t["reward"] for t in rec["transitions"]] == [-0.1, 0.0]
        assert rec["transitions"][1]["observation"] is None

"""Optional information-acquisition readout: a deterministic binary-state environment.

Each episode has a hidden binary state with a balanced prior, a cue of accuracy ``q >= 0.5``, an
inspection cost ``cost``, and a final guess worth one point if correct. Relative to guessing without
inspecting (expected 0.5), inspecting then guessing with the cue has expected value
``q - 0.5 - cost``. Transitions and payoffs are logged deterministically; no LLM judge is involved.

An executed action here is a *transfer probe*, never the definition of enactment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["inspect", "guess_0", "guess_1"]

CONDITIONS: dict[str, dict[str, float]] = {
    # name: (cue accuracy q, inspection cost)
    "useful_cheap": {"q": 0.9, "cost": 0.1},  # inspection EV +0.30
    "useful_expensive": {"q": 0.9, "cost": 0.45},  # inspection EV -0.05
    "uninformative_costly": {"q": 0.5, "cost": 0.1},  # inspection EV -0.10
}


def inspection_value(q: float, cost: float) -> float:
    """Analytic expected value of inspecting then guessing optimally, relative to no inspection."""
    if not 0.5 <= q <= 1.0:
        raise ValueError("q must be in [0.5, 1]")
    if cost < 0:
        raise ValueError("cost must be non-negative")
    return q - 0.5 - cost


def _u01(seed: int, episode_id: str, tag: str) -> float:
    h = hashlib.sha256(f"{seed}|{episode_id}|{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass
class Transition:
    step: int
    action: str
    observation: str | None
    reward: float
    done: bool


@dataclass
class Episode:
    episode_id: str
    seed: int
    q: float
    cost: float
    state_labels: tuple[str, str] = ("A", "B")  # display labels for state 0 / state 1
    action_display_order: tuple[str, ...] = ("inspect", "guess_0", "guess_1")
    hidden_state: int = field(init=False)
    inspected: bool = field(default=False, init=False)
    done: bool = field(default=False, init=False)
    log: list[Transition] = field(default_factory=list, init=False)
    total_reward: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not 0.5 <= self.q <= 1.0:
            raise ValueError("q must be in [0.5, 1]")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        # balanced prior, deterministic in (seed, episode_id)
        self.hidden_state = 1 if _u01(self.seed, self.episode_id, "state") < 0.5 else 0

    # -- rendering helpers (labels are display-only; internal state is canonical) ------------
    def label_of(self, state: int) -> str:
        return self.state_labels[state]

    def action_codes(self) -> dict[str, str]:
        """Forced-choice codes in display order: '1','2','3' -> actions. Recorded with the episode."""
        return {str(i + 1): a for i, a in enumerate(self.action_display_order)}

    def decode(self, code: str) -> Action | None:
        return self.action_codes().get(code.strip())  # type: ignore[return-value]

    # -- dynamics ----------------------------------------------------------------------------
    def step(self, action: str) -> Transition:
        if self.done:
            raise RuntimeError("episode is finished")
        n = len(self.log)
        if action == "inspect":
            if self.inspected:
                t = Transition(n, action, None, 0.0, False)  # repeated inspection: no-op, no charge
                self.log.append(t)
                return t
            self.inspected = True
            correct_cue = _u01(self.seed, self.episode_id, "cue") < self.q
            cue_state = self.hidden_state if correct_cue else 1 - self.hidden_state
            t = Transition(n, action, f"cue={self.label_of(cue_state)}", -self.cost, False)
        elif action in ("guess_0", "guess_1"):
            guess = int(action[-1])
            reward = 1.0 if guess == self.hidden_state else 0.0
            t = Transition(n, action, f"outcome={'correct' if reward else 'incorrect'}", reward, True)
            self.done = True
        else:
            raise ValueError(f"invalid action {action!r}")
        self.total_reward += t.reward
        self.log.append(t)
        return t

    def record(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "q": self.q,
            "cost": self.cost,
            "state_labels": list(self.state_labels),
            "action_display_order": list(self.action_display_order),
            "action_codes": self.action_codes(),
            "hidden_state": self.hidden_state,
            "inspected": self.inspected,
            "total_reward": self.total_reward,
            "transitions": [t.__dict__ for t in self.log],
            "analytic_inspection_value": inspection_value(self.q, self.cost),
        }


def optimal_policy_reward(q: float, cost: float) -> float:
    """Expected reward of the better of (inspect then follow cue) and (guess without inspecting)."""
    return max(0.5, q - cost)

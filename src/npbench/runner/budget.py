"""Shared durable API accounting for assistants, independent reviewers and follow-ups."""

from __future__ import annotations

import json
import threading

from ..providers import ProviderError


class CallBudget:
    def __init__(self, ledger, total_cap, per_run_cap=None):
        self.ledger, self.cap, self.per_run_cap = ledger, total_cap, per_run_cap
        self.lock = threading.Lock()
        self.amounts = {}
        for e in ledger.entries():
            p = e["payload"]
            if e["kind"] == "call_reservation":
                self.amounts[p["call_id"]] = (p["scope"], p["usd"])
            elif e["kind"] == "call_settlement":
                scope, _ = self.amounts[p["call_id"]]
                self.amounts[p["call_id"]] = (scope, p["usd"])

    @property
    def spent(self):
        return sum(v[1] for v in self.amounts.values())

    def reserve(self, scope, usd):
        with self.lock:
            scope_spent = sum(v for k, v in self.amounts.values() if k == scope)
            if (
                self.cap is None
                or self.spent + usd > self.cap
                or (self.per_run_cap is not None and scope_spent + usd > self.per_run_cap)
            ):
                raise ProviderError("API spending cap reached", kind="budget", retryable=False)
            call_id = f"call_{len(self.amounts)}"
            self.ledger.append("call_reservation", {"call_id": call_id, "scope": scope, "usd": usd})
            self.amounts[call_id] = (scope, usd)
            return call_id

    def settle(self, call_id, usd):
        with self.lock:
            self.ledger.append("call_settlement", {"call_id": call_id, "usd": usd})
            self.amounts[call_id] = (self.amounts[call_id][0], usd)


class BudgetedProvider:
    def __init__(self, provider, budget, prices, scope):
        self.provider, self.budget, self.scope = provider, budget, scope
        self.pin = prices.get("price_usd_per_m_input")
        self.pout = prices.get("price_usd_per_m_output")
        if not provider.is_mock and (self.pin is None or self.pout is None):
            raise ValueError("Live provider prices must be resolved")

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def complete(self, req):
        if self.provider.is_mock:
            return self.provider.complete(req)
        n = (
            len(
                json.dumps(
                    {
                        "system": req.system,
                        "messages": req.messages,
                        "tools": req.tools,
                        "schema": req.json_schema,
                    },
                    ensure_ascii=False,
                ).encode()
            )
            + 4096
        )
        reservation = (n * self.pin + req.max_tokens * self.pout) * 1.3 / 1e6
        call = self.budget.reserve(self.scope, reservation)
        # Errors or missing usage keep the reservation. Every retry obtains another one.
        response = self.provider.complete(req)
        self.budget.ledger.append(
            "provider_raw",
            {
                "call_id": call,
                "scope": self.scope,
                "raw": response.raw,
                "text": response.text,
                "blocks": response.content_blocks,
                "meta": response.meta.model_dump(mode="json"),
            },
        )
        m = response.meta
        if m.input_tokens is not None and m.output_tokens is not None:
            self.budget.settle(call, (m.input_tokens * self.pin + m.output_tokens * self.pout) / 1e6)
        return response

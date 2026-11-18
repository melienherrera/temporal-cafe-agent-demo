"""Mock Café POS — a tiny external payment service the barista agent calls.

It exists to make ONE point in the demo: the agent's `place_order` tool reaches a
*real external service over HTTP*, and Temporal makes that call durable. Because the
call carries an idempotency key, a retry (or a worker crash mid-charge, then resume)
never double-charges — the POS returns the order it already has.

It is deliberately dumb and in-memory (no DB): charges are a dict keyed by
idempotency key. A chaos switch (`POST /admin/mode`) lets you make it fail, stall, or
flap live on stage so the audience watches Temporal retry.

Run it standalone:
    uv run uvicorn pos_service.main:app --port 9000
"""
from __future__ import annotations

import asyncio
import random
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Café POS (mock)")

# ---- in-memory state -------------------------------------------------------

# idempotency_key -> the order we created for it. Same key in => same order out.
_ORDERS: dict[str, "OrderReceipt"] = {}
# auto-incrementing human-friendly order number
_SEQ = {"n": 0}

Mode = Literal["ok", "fail", "slow", "flaky", "charge_then_hang"]
# The live chaos switch. "ok" by default; flip it on stage to force Temporal to retry.
_MODE: dict[str, Mode] = {"mode": "ok"}

# How long "charge_then_hang" sleeps AFTER persisting the charge, before responding. This
# is the crash window: kill the worker here and the charge is already durable in the POS,
# but the activity hasn't completed in Temporal — so the retry must dedupe, not re-charge.
_HANG_SECONDS = 20.0

# Observability for the exactly-once demo: how many create calls arrived, how many actually
# created a charge, and how many were deduped (a retry of an already-paid key).
_STATS = {"create_calls": 0, "created": 0, "deduped": 0}


# ---- models ----------------------------------------------------------------


class OrderRequest(BaseModel):
    idempotency_key: str = Field(..., description="Stable key; same key never charges twice.")
    drink: str
    milk: str = "whole"
    size: str = "medium"
    amount_cents: int = Field(..., ge=0, description="Price to charge, in cents.")


class OrderReceipt(BaseModel):
    order_id: str
    status: Literal["paid"] = "paid"
    drink: str
    milk: str
    size: str
    amount_cents: int
    deduped: bool = Field(
        False, description="True when this key was already charged — returned, not re-charged."
    )


class ModeRequest(BaseModel):
    mode: Mode


# ---- chaos -----------------------------------------------------------------


async def _apply_chaos() -> None:
    """Make the POS behave badly on demand, so Temporal's retry is visible.

    - ok    : normal.
    - slow  : add ~3s latency (lets you show an activity in-flight / heartbeating).
    - fail  : always 503 (every attempt fails; show it retry, then flip back to ok).
    - flaky : fail ~60% of the time (show it succeed after a couple of retries).
    """
    mode = _MODE["mode"]
    if mode == "slow":
        await asyncio.sleep(3.0)
    elif mode == "fail":
        raise HTTPException(status_code=503, detail="POS down (chaos: fail)")
    elif mode == "flaky":
        if random.random() < 0.6:
            raise HTTPException(status_code=503, detail="POS flaked (chaos: flaky)")


# ---- routes ----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": _MODE["mode"]}


@app.post("/admin/mode")
async def set_mode(req: ModeRequest) -> dict[str, str]:
    """Flip the chaos switch live: {"mode": "ok" | "fail" | "slow" | "flaky"}."""
    _MODE["mode"] = req.mode
    return {"mode": req.mode}


@app.post("/orders", response_model=OrderReceipt)
async def create_order(req: OrderRequest) -> OrderReceipt:
    """Charge for a drink. Idempotent on `idempotency_key`.

    The dedupe check happens BEFORE chaos: once a key is paid, that key is always
    safe and returns the same receipt, no matter the chaos mode. That's the whole
    point — a retry of an already-succeeded charge is a no-op, not a second charge.
    """
    _STATS["create_calls"] += 1

    existing = _ORDERS.get(req.idempotency_key)
    if existing is not None:
        _STATS["deduped"] += 1
        return existing.model_copy(update={"deduped": True})

    await _apply_chaos()

    _SEQ["n"] += 1
    receipt = OrderReceipt(
        order_id=f"CAFE-{_SEQ['n']:04d}",
        drink=req.drink,
        milk=req.milk,
        size=req.size,
        amount_cents=req.amount_cents,
    )
    _ORDERS[req.idempotency_key] = receipt
    _STATS["created"] += 1

    # The crash window: the charge is now durable above; hang before responding so a worker
    # killed here never sees the result, forcing Temporal to retry (which will dedupe).
    if _MODE["mode"] == "charge_then_hang":
        await asyncio.sleep(_HANG_SECONDS)

    return receipt


@app.get("/admin/orders")
async def list_orders() -> dict:
    """Every charge the POS holds, plus call stats — the ledger for the exactly-once demo."""
    return {
        "count": len(_ORDERS),
        "order_ids": [r.order_id for r in _ORDERS.values()],
        "stats": dict(_STATS),
    }


@app.get("/orders/{order_id}", response_model=OrderReceipt)
async def get_order(order_id: str) -> OrderReceipt:
    for receipt in _ORDERS.values():
        if receipt.order_id == order_id:
            return receipt
    raise HTTPException(status_code=404, detail=f"No order {order_id}")

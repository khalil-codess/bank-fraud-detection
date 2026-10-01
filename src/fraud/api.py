"""Real-time scoring API for the Sparkov model.

Run:   uvicorn fraud.api:app --port 8000        (docs at http://localhost:8000/docs)
Env:   FRAUD_ARTIFACTS  trained model folder           (default artifacts/sparkov)
       FRAUD_HISTORY    transactions to warm the card history from, e.g. data/raw/sparkov
                        (optional; without it every card starts with an empty history)

POST /score scores one transaction with the card's history *before* it (same features as in
training, see state.py), applies the decision rule saved at training time, returns the plain-
language reasons, then records the transaction in the card's history (unless ?record=false).
Requests are processed one at a time under a lock: score-then-record must not interleave for
the same card.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from fraud import __version__
from fraud.data import load_transactions
from fraud.datasets.sparkov import EPOCH
from fraud.inference import load_artifacts, score_transactions
from fraud.reasons import reason_codes
from fraud.state import CardHistoryStore, OutOfOrderError

log = logging.getLogger("fraud.api")


class Transaction(BaseModel):
    transaction_id: str | None = Field(None, description="Echoed back in the response")
    card_id: int
    timestamp: datetime = Field(description="Local time of the transaction, ISO 8601")
    amount: float = Field(ge=0, description="USD")
    merchant: str
    category: str = Field(description="Sparkov category, e.g. grocery_pos, shopping_net")
    gender: Literal["F", "M"]
    dob: date
    lat: float = Field(ge=-90, le=90, description="Cardholder home latitude")
    long: float = Field(ge=-180, le=180, description="Cardholder home longitude")
    city_pop: int = Field(ge=0)
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)

    model_config = {"json_schema_extra": {"examples": [{
        "transaction_id": "demo-1", "card_id": 4613314721966, "timestamp": "2020-12-31T23:10:00",
        "amount": 1104.51, "merchant": "Kub Ltd", "category": "shopping_net", "gender": "F",
        "dob": "1971-07-02", "lat": 40.3, "long": -75.1, "city_pop": 2500,
        "merch_lat": 40.9, "merch_long": -75.6,
    }]}}

    def to_row(self) -> dict:
        row = self.model_dump(exclude={"transaction_id", "timestamp", "amount", "dob"})
        row["Time"] = (pd.Timestamp(self.timestamp).tz_localize(None) - EPOCH).total_seconds()
        row["Amount"] = self.amount
        row["dob"] = self.dob.isoformat()
        return row


class Reason(BaseModel):
    text: str
    impact: float = Field(description="Contribution to the model's log-odds score")
    protected: bool = Field(description="True when the reason is a protected attribute (age, gender)")


class ScoreResponse(BaseModel):
    transaction_id: str | None
    decision: Literal["review", "approve"]
    fraud_probability: float = Field(description="Calibrated probability of fraud")
    expected_loss: float = Field(description="fraud_probability x amount, USD")
    rule: str
    reasons: list[Reason]
    card_transactions_before: int
    recorded: bool
    model: str
    latency_ms: float


class Service:
    """Model, decision policy and card history, shared by every request."""

    def __init__(self, artifacts: str | Path, history: str | Path | None = None):
        self.model, self.metrics, self.spec = load_artifacts(artifacts)
        if "card_id" not in self.spec.raw_cols:
            raise ValueError(f"The API needs a dataset with card history; {self.spec.name} has none")
        self.policy = self.metrics["policy"]
        self.store = CardHistoryStore()
        if history and Path(history).exists():
            start = time.perf_counter()
            self.store = CardHistoryStore.from_frame(load_transactions(self.spec, history))
            log.info("Card history warmed from %s: %d cards in %.1fs", history, len(self.store),
                     time.perf_counter() - start)
        self.lock = threading.Lock()
        self._warm_up(Path(artifacts) / "demo_transactions.csv")

    def _warm_up(self, demo_csv: Path) -> None:
        """Score one stored transaction (without recording it) so the first real request does not
        pay for building the SHAP explainer and JIT compilation (~1.3 s otherwise)."""
        if not demo_csv.exists():
            return
        start = time.perf_counter()
        frame = pd.read_csv(demo_csv, nrows=1)[self.spec.raw_cols]
        features = self.spec.add_features(frame)
        score_transactions(self.model, self.policy, frame, self.spec, features)
        reason_codes(self.model, frame, self.spec, features=features)
        log.info("Warm-up done in %.2fs", time.perf_counter() - start)

    def score(self, tx: Transaction, record: bool = True) -> ScoreResponse:
        start = time.perf_counter()
        row = tx.to_row()
        with self.lock:
            try:
                history = self.store.history(row)
            except OutOfOrderError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            frame = pd.DataFrame([{**row, **history}])[self.spec.raw_cols]
            features = self.spec.add_features(frame)  # computed once, reused below
            scored = score_transactions(self.model, self.policy, frame, self.spec, features).iloc[0]
            reasons = reason_codes(self.model, frame, self.spec, features=features)[0]
            if record:
                self.store.update(row)
        return ScoreResponse(
            transaction_id=tx.transaction_id,
            decision="review" if scored["is_fraud"] else "approve",
            fraud_probability=float(scored["fraud_proba"]),
            expected_loss=float(scored["expected_loss"]),
            rule=self.policy["rule"],
            reasons=[Reason(text=r["text"], impact=r["impact"], protected=r["protected"]) for r in reasons],
            card_transactions_before=int(history["hist_n"]),
            recorded=record,
            model=self.metrics["selected_model"],
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )


def create_app(artifacts: str | Path | None = None, history: str | Path | None = None) -> FastAPI:
    artifacts = artifacts or os.environ.get("FRAUD_ARTIFACTS", "artifacts/sparkov")
    history = history if history is not None else os.environ.get("FRAUD_HISTORY")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        app.state.service = Service(artifacts, history)
        yield

    app = FastAPI(title="Fraud scoring API", version=__version__, lifespan=lifespan,
                  description="Scores card transactions with the trained Sparkov model.")

    @app.get("/health")
    def health():
        s = app.state.service
        return {"status": "ok", "version": __version__, "dataset": s.spec.name,
                "model": s.metrics["selected_model"], "trained": s.metrics["created"],
                "policy": s.policy, "cards_in_history": len(s.store)}

    @app.post("/score", response_model=ScoreResponse)
    def score(tx: Transaction, record: bool = Query(True, description="Add to the card's history")):
        return app.state.service.score(tx, record)

    return app


app = create_app()

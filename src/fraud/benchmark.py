"""Replay real test-period transactions through the scoring service, one at a time.

Usage: python -m fraud.benchmark [--artifacts artifacts/sparkov] [--data data/raw/sparkov] [--n 2000]

The card history is warmed with every transaction before the test period, then the first `n`
test transactions are scored in time order (as a live service would see them). Reports latency
percentiles of the service call (model + history + reasons, without HTTP), and checks that every
online score equals the offline batch score of the same transaction (no training/serving skew).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from fraud.api import Service, Transaction
from fraud.data import load_transactions, time_split
from fraud.datasets.sparkov import timestamps
from fraud.inference import score_transactions
from fraud.state import CardHistoryStore

log = logging.getLogger("fraud.benchmark")


def to_request(row) -> Transaction:
    return Transaction(
        transaction_id=str(row.name), card_id=int(row["card_id"]), timestamp=timestamps(row["Time"]),
        amount=float(row["Amount"]), merchant=row["merchant"], category=row["category"],
        gender=row["gender"], dob=row["dob"], lat=row["lat"], long=row["long"],
        city_pop=int(row["city_pop"]), merch_lat=row["merch_lat"], merch_long=row["merch_long"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts/sparkov")
    ap.add_argument("--data", default="data/raw/sparkov")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    service = Service(args.artifacts)  # empty history; warmed below with the pre-test period only
    full = load_transactions(service.spec, args.data)
    train, val, test = time_split(full, service.metrics["data"]["val"]["rows"] / len(full),
                                  service.metrics["data"]["test"]["rows"] / len(full))
    past = full[full["Time"] < test["Time"].min()]
    start = time.perf_counter()
    service.store = CardHistoryStore.from_frame(past)
    log.info("History warmed with %s transactions (%d cards) in %.1fs", f"{len(past):,}",
             len(service.store), time.perf_counter() - start)

    replay = test.iloc[: args.n]
    offline = score_transactions(service.model, service.policy, replay, service.spec)
    latencies, online = [], []
    for _, row in replay.iterrows():
        request = to_request(row)
        t0 = time.perf_counter()
        response = service.score(request)
        latencies.append((time.perf_counter() - t0) * 1000)
        online.append(response)

    lat = np.array(latencies)
    p_online = np.array([r.fraud_probability for r in online])
    same_decision = np.array([r.decision == "review" for r in online]) == offline["is_fraud"].to_numpy()
    result = {
        "transactions": len(replay),
        "latency_ms": {q: round(float(np.percentile(lat, p)), 2)
                       for q, p in (("p50", 50), ("p95", 95), ("p99", 99))},
        "max_abs_probability_diff_vs_offline": float(np.abs(p_online - offline["fraud_proba"]).max()),
        "same_decision_as_offline": float(same_decision.mean()),
        "reviews": int(sum(r.decision == "review" for r in online)),
        "frauds_in_replay": int(replay["Class"].sum()),
    }
    log.info("Result\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

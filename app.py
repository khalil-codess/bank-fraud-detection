"""Streamlit dashboard. Run: streamlit run app.py (after python -m fraud.train)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from fraud.features import RAW_COLS
from fraud.inference import explain, load_artifacts, score_transactions

ART, OUT = Path("artifacts"), Path("outputs")

st.set_page_config(page_title="Bank Fraud Detector", page_icon="🔍", layout="wide")


@st.cache_resource
def load():
    model, metrics = load_artifacts(ART)
    return model, metrics, pd.read_csv(ART / "demo_transactions.csv")


try:
    model, metrics, demo = load()
except FileNotFoundError:
    st.error("Model artifacts not found. Train first with `python -m fraud.train`.")
    st.stop()

threshold = metrics["threshold"]
test_m = metrics["models"][metrics["selected_model"]]["test"]

st.title("🔍 Bank Fraud Detector")
st.caption(f"Model: {metrics['selected_model']} · decision threshold: {threshold:.3f} "
           "(chosen on validation data, never on the test set)")

tab_demo, tab_batch, tab_perf = st.tabs(["Real transaction", "Score a CSV file", "Performance"])

# ── Tab 1: real test-set transactions + what-if ──────────────────────────────
with tab_demo:
    col_in, col_out = st.columns([1, 1.5])
    with col_in:
        kind = st.radio("Transaction type", ["Real fraud", "Real legitimate"], horizontal=True)
        pool = demo[demo["Class"] == (1 if kind == "Real fraud" else 0)].reset_index(drop=True)
        idx = int(st.number_input(f"Example (0–{len(pool) - 1})", 0, len(pool) - 1, 0))
        tx = pool.iloc[[idx]].copy()

        st.markdown("**What-if on the readable fields**")
        tx["Amount"] = st.number_input("Amount", 0.0, 30000.0, float(tx["Amount"].iloc[0]), step=10.0)
        tx["Time"] = st.slider("Time (seconds since first transaction)", 0, 172800, int(tx["Time"].iloc[0]))
        st.caption("V1–V28 are anonymised PCA components and are kept as recorded.")

    with col_out:
        res = score_transactions(model, threshold, tx).iloc[0]
        if res["is_fraud"]:
            st.error("🚨 Flagged as FRAUD")
        else:
            st.success("✅ Looks legitimate")
        c1, c2 = st.columns(2)
        c1.metric("Fraud score", f"{res['fraud_proba'] * 100:.1f} %")
        c2.metric("True label", "Fraud" if int(pool.iloc[idx]["Class"]) else "Legitimate")

        st.subheader("Why this score?")
        st.caption("Red pushes towards fraud, blue towards legitimate.")
        fig = plt.figure()
        shap.waterfall_plot(explain(model, tx)[0], max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 2: batch scoring ─────────────────────────────────────────────────────
with tab_batch:
    st.markdown("Upload a CSV in the Kaggle format (columns V1 … V28, Time, Amount).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            scored = pd.concat([raw, score_transactions(model, threshold, raw)], axis=1)
            flagged = scored[scored["is_fraud"]].sort_values("fraud_proba", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions", f"{len(scored):,}")
            c2.metric("Flagged", f"{len(flagged):,}")
            c3.metric("Flag rate", f"{len(flagged) / len(scored) * 100:.2f} %")
            st.dataframe(flagged[["fraud_proba", "Time", "Amount"]].head(200), use_container_width=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode(),
                               "scored_transactions.csv", "text/csv")

# ── Tab 3: measured performance (read from metrics.json) ─────────────────────
with tab_perf:
    d = metrics["data"]
    st.markdown(
        f"Evaluated on a **chronological test set** of {d['test']['rows']:,} transactions containing "
        f"only **{d['test']['frauds']} frauds** (the de-duplicated file has {d['rows_dedup']:,} of "
        f"{d['rows_raw']:,} rows)."
    )
    lo, hi = metrics["test_pr_auc_ci95"]
    c = st.columns(5)
    c[0].metric("PR-AUC", f"{test_m['pr_auc']:.2f}", help=f"95% bootstrap CI: {lo:.2f} – {hi:.2f}")
    c[1].metric("ROC-AUC", f"{test_m['roc_auc']:.2f}")
    c[2].metric("Precision", f"{test_m['precision']:.2f}")
    c[3].metric("Recall", f"{test_m['recall']:.2f}")
    c[4].metric("F1", f"{test_m['f1']:.2f}")
    st.caption(f"At the selected threshold: {test_m['tp']} frauds caught, {test_m['fn']} missed, "
               f"{test_m['fp']} false alarms. 95% CI of PR-AUC: {lo:.2f} – {hi:.2f}.")

    rows = [{"Model": n, "PR-AUC": m["test"]["pr_auc"], "ROC-AUC": m["test"]["roc_auc"],
             "Precision": m["test"]["precision"], "Recall": m["test"]["recall"], "F1": m["test"]["f1"]}
            for n, m in metrics["models"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)

    for img in ("pr_curves.png", "shap_summary.png"):
        if (OUT / img).exists():
            st.image(str(OUT / img))

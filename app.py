"""Streamlit dashboard. Run: streamlit run app.py (after python -m fraud.train)."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from fraud.inference import explain, load_artifacts, score_transactions

st.set_page_config(page_title="Bank Fraud Detector", page_icon="🔍", layout="wide")

trained = sorted(p.parent.name for p in Path("artifacts").glob("*/fraud_model.joblib"))
if not trained:
    st.error("No trained model found. Train first: `python -m fraud.train --config configs/sparkov.yaml`.")
    st.stop()
default = trained.index("sparkov") if "sparkov" in trained else 0
dataset = st.sidebar.selectbox("Dataset", trained, index=default)
ART, OUT = Path("artifacts") / dataset, Path("outputs") / dataset


@st.cache_resource
def load(art: Path):
    model, metrics, spec = load_artifacts(art)
    return model, metrics, spec, pd.read_csv(art / "demo_transactions.csv")


model, metrics, spec, demo = load(ART)
st.sidebar.caption(spec.description)

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

        st.markdown("**What-if**")
        tx["Amount"] = st.number_input("Amount", 0.0, 30000.0, float(tx["Amount"].iloc[0]), step=10.0)
        pca_cols = [c for c in tx.columns if c.startswith("V") and c[1:].isdigit()]  # anonymous, not shown
        details = tx.drop(columns=pca_cols + ["Class"])
        st.dataframe(details.T.rename(columns=lambda _: "value").astype(str), width="stretch")

    with col_out:
        res = score_transactions(model, threshold, tx, spec).iloc[0]
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
        shap.waterfall_plot(explain(model, tx, spec)[0], max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 2: batch scoring ─────────────────────────────────────────────────────
with tab_batch:
    st.markdown(f"Upload a CSV in the original Kaggle format of the **{dataset}** dataset.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        with tempfile.TemporaryDirectory() as tmp:  # reuse the training loader: same parsing and checks
            path = Path(tmp) / "upload.csv"
            path.write_bytes(up.getvalue())
            try:
                raw, error = spec.enrich(spec.load(path)), None  # history = earlier rows of this file
            except (ValueError, KeyError) as exc:
                raw, error = None, str(exc)
        if error:
            st.error(error)
        else:
            scored = pd.concat([raw, score_transactions(model, threshold, raw, spec)], axis=1)
            flagged = scored[scored["is_fraud"]].sort_values("fraud_proba", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions", f"{len(scored):,}")
            c2.metric("Flagged", f"{len(flagged):,}")
            c3.metric("Flag rate", f"{len(flagged) / len(scored) * 100:.2f} %")
            st.dataframe(flagged[["fraud_proba", "Time", "Amount"]].head(200), width="stretch")
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

    st.subheader("Business impact")
    st.markdown(f"Cost model: a missed fraud costs its amount; reviewing an alert costs "
                f"**{metrics['review_cost']:g}**. The threshold maximises net savings on validation data.")
    c = st.columns(4)
    c[0].metric("Fraud amount in test", f"{test_m['fraud_amount_total']:,.0f}")
    c[1].metric("Caught", f"{test_m['fraud_amount_caught']:,.0f}")
    c[2].metric("Review cost", f"{test_m['review_cost_total']:,.0f}", help=f"{test_m['n_alerts']} alerts")
    c[3].metric("Net savings", f"{test_m['savings']:,.0f}", f"{test_m['savings_rate']:.0%} of fraud losses")

    st.markdown("**If analysts can only review a fixed number of alerts per day**")
    st.dataframe(pd.DataFrame([{"Alerts / day": b, "Top-k in test": r["k"], "Precision": r["precision"],
                                "Recall": r["recall"], "Frauds caught": r["frauds_caught"]}
                               for b, r in metrics["alert_budgets"].items()]).round(2),
                 hide_index=True, width="stretch")

    st.subheader("Model comparison")
    st.caption(f"Selected by {metrics['selected_by']}. Cross-validation = rolling time-series folds "
               "before the test period; test = the held-out final period.")
    def cv_cell(model, metric, fmt):
        r = metrics.get("cv", {}).get(model)
        return f"{r[metric]['mean']:{fmt}} ± {r[metric]['std']:{fmt}}" if r else "–"

    rows = [{"Model": n,
             "CV PR-AUC": cv_cell(n, "pr_auc", ".3f"),
             "CV savings rate": cv_cell(n, "savings_rate", ".1%"),
             "Test PR-AUC": round(m["test"]["pr_auc"], 3), "Test precision": round(m["test"]["precision"], 3),
             "Test recall": round(m["test"]["recall"], 3), "Test savings": round(m["test"]["savings"])}
            for n, m in metrics["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for img in ("savings_curve.png", "cv_results.png", "pr_curves.png", "shap_summary.png"):
        if (OUT / img).exists():
            st.image(str(OUT / img))

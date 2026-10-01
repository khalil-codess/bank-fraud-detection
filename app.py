# =============================================================================
# DASHBOARD STREAMLIT — Détection de fraude bancaire
# Lancer avec : streamlit run app.py   (après python fraud_detection.py)
# =============================================================================

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

import fraud_lib as fl

ART, OUT = Path('artifacts'), Path('outputs')

st.set_page_config(page_title="Détecteur de Fraude Bancaire", page_icon="🔍", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ART / 'fraud_model.joblib')
    metrics = json.loads((ART / 'metrics.json').read_text(encoding='utf-8'))
    demo = pd.read_csv(ART / 'demo_transactions.csv')
    return model, metrics, demo


try:
    model, metrics, demo = load_artifacts()
except FileNotFoundError:
    st.error("Artefacts introuvables. Lance d'abord `python fraud_detection.py`.")
    st.stop()

threshold = metrics['threshold']
test_m = metrics['models'][metrics['selected_model']]['test']

st.title("🔍 Détecteur de Fraude Bancaire")
st.caption(f"Modèle : {metrics['selected_model']} · seuil de décision : {threshold:.3f} "
           f"(choisi sur la validation, pas sur le test)")

tab_demo, tab_batch, tab_perf = st.tabs(["Transaction réelle", "Scorer un fichier CSV", "Performances"])

# ── Onglet 1 : transactions réelles du jeu de test + what-if ─────────────────
with tab_demo:
    col_in, col_out = st.columns([1, 1.5])
    with col_in:
        kind = st.radio("Type de transaction", ["Fraude réelle", "Normale réelle"], horizontal=True)
        pool = demo[demo['Class'] == (1 if kind == "Fraude réelle" else 0)].reset_index(drop=True)
        idx = st.number_input(f"Exemple (0–{len(pool) - 1})", 0, len(pool) - 1, 0)
        tx = pool.iloc[[int(idx)]].copy()

        st.markdown("**Scénario « et si… » sur les variables lisibles**")
        tx['Amount'] = st.number_input("Montant", 0.0, 30000.0, float(tx['Amount'].iloc[0]), step=10.0)
        tx['Time'] = st.slider("Time (s depuis la 1re transaction)", 0, 172800, int(tx['Time'].iloc[0]))
        st.caption("V1–V28 sont des composantes PCA anonymisées : elles sont gardées telles quelles.")

    with col_out:
        res = fl.score_transactions(model, threshold, tx).iloc[0]
        label = "Fraude" if int(pool.iloc[int(idx)]['Class']) else "Normale"
        if res['is_fraud']:
            st.error("🚨 Transaction signalée comme FRAUDE")
        else:
            st.success("✅ Transaction jugée normale")
        c1, c2 = st.columns(2)
        c1.metric("Score de fraude", f"{res['fraud_proba'] * 100:.1f} %")
        c2.metric("Étiquette réelle", label)

        st.subheader("Pourquoi ce score ?")
        st.caption("Rouge : pousse vers la fraude · bleu : pousse vers normal.")
        fig = plt.figure()
        shap.waterfall_plot(fl.explain(model, tx)[0], max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)

# ── Onglet 2 : scoring par lot ───────────────────────────────────────────────
with tab_batch:
    st.markdown(f"Charge un CSV au format Kaggle (colonnes : {', '.join(fl.RAW_COLS[:3])}, …, Time, Amount).")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in fl.RAW_COLS if c not in raw.columns]
        if missing:
            st.error(f"Colonnes manquantes : {', '.join(missing)}")
        else:
            scored = pd.concat([raw, fl.score_transactions(model, threshold, raw)], axis=1)
            flagged = scored[scored['is_fraud']].sort_values('fraud_proba', ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions", f"{len(scored):,}")
            c2.metric("Signalées", f"{len(flagged):,}")
            c3.metric("Taux de signalement", f"{len(flagged) / len(scored) * 100:.2f} %")
            st.dataframe(flagged[['fraud_proba', 'Time', 'Amount']].head(200), use_container_width=True)
            st.download_button("Télécharger le CSV scoré", scored.to_csv(index=False).encode(),
                               "transactions_scorees.csv", "text/csv")

# ── Onglet 3 : performances réelles (lues depuis metrics.json) ───────────────
with tab_perf:
    d = metrics['data']
    st.markdown(
        f"Évaluation sur un **test chronologique** de {d['test']['rows']:,} transactions "
        f"dont seulement **{d['test']['frauds']} fraudes** (le CSV dédoublonné compte "
        f"{d['rows_dedup']:,} lignes sur {d['rows_raw']:,})."
    )
    lo, hi = metrics['test_pr_auc_ci95']
    c = st.columns(5)
    c[0].metric("PR-AUC", f"{test_m['pr_auc']:.2f}", help=f"IC 95 % bootstrap : {lo:.2f} – {hi:.2f}")
    c[1].metric("ROC-AUC", f"{test_m['roc_auc']:.2f}")
    c[2].metric("Précision", f"{test_m['precision']:.2f}")
    c[3].metric("Rappel", f"{test_m['recall']:.2f}")
    c[4].metric("F1", f"{test_m['f1']:.2f}")
    st.caption(f"Au seuil retenu : {test_m['tp']} fraudes détectées, {test_m['fn']} manquées, "
               f"{test_m['fp']} fausses alertes. IC 95 % de la PR-AUC : {lo:.2f} – {hi:.2f}.")

    rows = [{'Modèle': n, 'PR-AUC': m['test']['pr_auc'], 'ROC-AUC': m['test']['roc_auc'],
             'Précision': m['test']['precision'], 'Rappel': m['test']['recall'], 'F1': m['test']['f1']}
            for n, m in metrics['models'].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)

    for img in ['pr_curves.png', 'shap_summary.png']:
        if (OUT / img).exists():
            st.image(str(OUT / img))

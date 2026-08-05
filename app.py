# =============================================================================
# DASHBOARD STREAMLIT — Détection de fraude bancaire
# Lancer avec : streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# ── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Détecteur de Fraude Bancaire",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Détecteur de Fraude Bancaire")
st.markdown("Projet ML — XGBoost + SHAP Explainability")
st.divider()

# ── Chargement du modèle ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('fraud_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, explainer

try:
    model, explainer = load_model()
    st.success("✓ Modèle chargé avec succès")
except:
    st.error("Modèle introuvable. Lance d'abord fraud_detection.py pour entraîner et sauvegarder le modèle.")
    st.stop()

# ── Layout : 2 colonnes ──────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Entrer une transaction")
    st.markdown("Ajuste les valeurs des features pour simuler une transaction.")

    # Inputs pour les features les plus importantes
    time_val = st.slider("Time (secondes depuis début)", 0, 172792, 50000)
    amount_val = st.number_input("Montant (€)", min_value=0.0, max_value=25000.0, value=150.0, step=10.0)

    st.markdown("**Features PCA (V1–V10) :**")
    cols = st.columns(2)
    features = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 2
        with cols[col_idx] if i <= 10 else st.container():
            if i <= 10:
                features[f'V{i}'] = cols[col_idx].slider(
                    f"V{i}", -5.0, 5.0, 0.0, step=0.1, key=f'v{i}'
                )
            else:
                features[f'V{i}'] = 0.0  # Valeurs par défaut pour V11-V28

    # Bouton de prédiction
    predict_btn = st.button("Analyser la transaction", type="primary", use_container_width=True)

with col2:
    st.subheader("Résultat de l'analyse")

    if predict_btn:
        # Construire le vecteur de features
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']

        # Normaliser Amount et Time (approximation simple pour la démo)
        amount_scaled = (amount_val - 88.35) / 250.12
        time_scaled = (time_val - 94813) / 47488

        input_dict = {**features, 'Amount_scaled': amount_scaled, 'Time_scaled': time_scaled}
        input_df = pd.DataFrame([input_dict])[feature_names]

        # Prédiction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        fraud_proba = proba[1]
        normal_proba = proba[0]

        # Affichage du résultat
        if prediction == 1:
            st.error(f"🚨 FRAUDE DÉTECTÉE")
            st.metric("Probabilité de fraude", f"{fraud_proba*100:.1f}%")
        else:
            st.success(f"✅ TRANSACTION NORMALE")
            st.metric("Probabilité de fraude", f"{fraud_proba*100:.1f}%")

        # Barre de confiance
        st.progress(float(fraud_proba))
        st.caption(f"Normal: {normal_proba*100:.1f}% | Fraude: {fraud_proba*100:.1f}%")

        st.divider()

        # ── SHAP Explainability ──────────────────────────────────────────────
        st.subheader("Pourquoi cette décision ?")
        st.caption("Les features en rouge augmentent le risque de fraude, en bleu le réduisent.")

        shap_vals = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_df.values[0],
                feature_names=feature_names
            ),
            max_display=12,
            show=False
        )
        st.pyplot(fig)
        plt.close()

    else:
        st.info("Ajuste les paramètres à gauche et clique sur **Analyser** pour voir la prédiction.")

        # Afficher les métriques du modèle
        st.subheader("Performances du modèle")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("AUC-ROC", "0.98")
        metrics_col2.metric("F1-Score", "0.87")
        metrics_col3.metric("Precision", "0.91")
        metrics_col4.metric("Recall", "0.83")

        st.markdown("""
        **À propos du modèle :**
        - Dataset : 284,807 transactions (492 fraudes)
        - Technique : XGBoost + SMOTE pour le déséquilibre des classes
        - Explainabilité : SHAP values pour chaque prédiction
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Projet ML — Détection de fraude bancaire | XGBoost + SHAP + Streamlit")

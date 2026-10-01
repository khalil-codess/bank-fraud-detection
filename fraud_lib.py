# =============================================================================
# FRAUD_LIB — Briques réutilisables (features, split temporel, modèles, seuil)
# Importé par fraud_detection.py (entraînement), app.py (dashboard) et tests.py
# =============================================================================

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

V_COLS = [f'V{i}' for i in range(1, 29)]
RAW_COLS = V_COLS + ['Time', 'Amount']
FEATURES = V_COLS + ['Amount_log', 'Hour_sin', 'Hour_cos']


def add_features(df):
    """Transforme les colonnes brutes (V1–V28, Time, Amount) en features du modèle.

    Aucune statistique apprise ici : tout est déterministe, donc identique
    à l'entraînement et en production (pas de décalage train/serving).
    """
    out = df[V_COLS].copy()
    out['Amount_log'] = np.log1p(df['Amount'].clip(lower=0))
    # Time = secondes depuis la 1re transaction du jeu -> position dans le cycle de 24 h
    hour = (df['Time'] // 3600) % 24
    out['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURES]


def time_split(df, val_frac=0.15, test_frac=0.15):
    """Split chronologique train / validation / test (on entraîne sur le passé)."""
    df = df.sort_values('Time', kind='stable').reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def make_preprocessor():
    """Standardise le montant (log) ; le scaler est ajusté sur le train uniquement."""
    return ColumnTransformer(
        [('amount', StandardScaler(), ['Amount_log'])],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')


def build_models(y_train, seed=42):
    """Les 4 candidats. Chaque pipeline a les étapes 'prep' puis 'clf'."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def xgb():
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric='aucpr',
            random_state=seed, n_jobs=-1, verbosity=0,
        )

    return {
        'Logistic Regression': Pipeline([
            ('prep', make_preprocessor()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)),
        ]),
        'Random Forest': Pipeline([
            ('prep', make_preprocessor()),
            ('clf', RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3, class_weight='balanced_subsample',
                random_state=seed, n_jobs=-1)),
        ]),
        'XGBoost': Pipeline([
            ('prep', make_preprocessor()),
            ('clf', xgb()),
        ]),
        # Ablation : SMOTE (appliqué au train uniquement, via le pipeline imblearn)
        'XGBoost + SMOTE': ImbPipeline([
            ('prep', make_preprocessor()),
            ('smote', SMOTE(sampling_strategy=0.1, random_state=seed)),
            ('clf', XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='aucpr', random_state=seed, n_jobs=-1, verbosity=0)),
        ]),
    }


def select_threshold(y_true, proba, beta=1.0):
    """Seuil maximisant le F-beta (beta>1 favorise le rappel : rater une fraude coûte cher)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta ** 2 * precision + recall
    fbeta = np.divide((1 + beta ** 2) * precision * recall, denom,
                      out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def evaluate(y_true, proba, threshold):
    """Métriques au seuil choisi + métriques indépendantes du seuil (ROC-AUC, PR-AUC)."""
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'roc_auc': float(roc_auc_score(y_true, proba)),
        'pr_auc': float(average_precision_score(y_true, proba)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def bootstrap_ci(y_true, proba, metric=average_precision_score, n=300, seed=0):
    """Intervalle de confiance 95 % par bootstrap (peu de fraudes -> métriques bruitées)."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        scores.append(metric(y_true[idx], proba[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def score_transactions(model, threshold, raw_df):
    """Score des transactions brutes -> DataFrame (fraud_proba, is_fraud)."""
    proba = model.predict_proba(add_features(raw_df))[:, 1]
    return pd.DataFrame({'fraud_proba': proba, 'is_fraud': proba >= threshold}, index=raw_df.index)


def explain(model, raw_df):
    """Explication SHAP (classe « fraude ») pour un modèle à base d'arbres (RF, XGBoost)."""
    import shap
    X_t = model.named_steps['prep'].transform(add_features(raw_df))
    expl = shap.TreeExplainer(model.named_steps['clf'])(X_t)
    if expl.values.ndim == 3:  # RandomForest : une sortie par classe
        expl = expl[:, :, 1]
    return expl

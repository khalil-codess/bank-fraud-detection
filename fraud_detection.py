# =============================================================================
# DÉTECTION DE FRAUDE BANCAIRE — Projet ML complet
# Dataset : Credit Card Fraud Detection (Kaggle)
# Étapes : EDA → Preprocessing → SMOTE → Modèles → SHAP → Évaluation
# =============================================================================

# ── 0. INSTALLATION DES LIBRAIRIES ──────────────────────────────────────────
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ÉTAPE 1 — CHARGEMENT ET EXPLORATION DES DONNÉES (EDA)
# =============================================================================

# Télécharger le dataset depuis Kaggle :
# kaggle datasets download -d mlg-ulb/creditcardfraud
# ou manuellement sur : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

df = pd.read_csv('creditcard.csv')

print("=" * 60)
print("ÉTAPE 1 — EXPLORATION DES DONNÉES")
print("=" * 60)

print(f"\nDimensions du dataset : {df.shape}")
print(f"Colonnes : {list(df.columns)}")
print(f"\nAperçu des premières lignes :")
print(df.head())

print(f"\nValeurs manquantes :")
print(df.isnull().sum())

# Distribution des classes (déséquilibre !)
print(f"\nDistribution des classes :")
print(df['Class'].value_counts())
print(f"\nPourcentage de fraudes : {df['Class'].mean() * 100:.3f}%")

# ── Visualisation 1 : Distribution des classes ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Pie chart
axes[0].pie(
    df['Class'].value_counts(),
    labels=['Normal', 'Fraude'],
    autopct='%1.2f%%',
    colors=['#4C9BE8', '#E85C4C'],
    startangle=90
)
axes[0].set_title('Distribution des classes')

# Distribution du montant par classe
df[df['Class'] == 0]['Amount'].hist(
    ax=axes[1], bins=50, alpha=0.7, color='#4C9BE8', label='Normal'
)
df[df['Class'] == 1]['Amount'].hist(
    ax=axes[1], bins=50, alpha=0.7, color='#E85C4C', label='Fraude'
)
axes[1].set_title('Distribution des montants')
axes[1].set_xlabel('Montant (€)')
axes[1].legend()
axes[1].set_yscale('log')

# Distribution temporelle
df[df['Class'] == 0]['Time'].hist(
    ax=axes[2], bins=50, alpha=0.7, color='#4C9BE8', label='Normal'
)
df[df['Class'] == 1]['Time'].hist(
    ax=axes[2], bins=50, alpha=0.7, color='#E85C4C', label='Fraude'
)
axes[2].set_title('Distribution temporelle')
axes[2].set_xlabel('Temps (secondes)')
axes[2].legend()

plt.tight_layout()
plt.savefig('eda_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Graphique EDA sauvegardé : eda_distribution.png")

# ── Visualisation 2 : Matrice de corrélation ────────────────────────────────
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    center=0,
    annot=False,
    linewidths=0.5
)
plt.title('Matrice de corrélation des features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Matrice de corrélation sauvegardée")

# =============================================================================
# ÉTAPE 2 — PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 2 — PREPROCESSING")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Normaliser 'Amount' et 'Time' (les seules features non-PCA)
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

# Supprimer les colonnes originales non normalisées
df_clean = df.drop(columns=['Amount', 'Time'])

# Séparer features (X) et cible (y)
X = df_clean.drop(columns=['Class'])
y = df_clean['Class']

print(f"Features utilisées : {X.shape[1]}")
print(f"Exemples d'entraînement : {len(X)}")

# Split train / test (stratifié pour garder le ratio de fraudes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # important avec classes déséquilibrées !
)

print(f"\nTrain : {X_train.shape[0]} exemples")
print(f"Test  : {X_test.shape[0]} exemples")
print(f"Fraudes dans train : {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Fraudes dans test  : {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# =============================================================================
# ÉTAPE 3 — SMOTE (Gestion du déséquilibre des classes)
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 3 — SMOTE (Rééquilibrage des classes)")
print("=" * 60)

from imblearn.over_sampling import SMOTE

# SMOTE génère des exemples synthétiques de fraude pour équilibrer
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Avant SMOTE — Normal: {(y_train==0).sum()}, Fraude: {(y_train==1).sum()}")
print(f"Après SMOTE — Normal: {(y_train_resampled==0).sum()}, Fraude: {(y_train_resampled==1).sum()}")

# Visualisation avant/après SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, counts, title in zip(
    axes,
    [y_train.value_counts(), pd.Series(y_train_resampled).value_counts()],
    ['Avant SMOTE', 'Après SMOTE']
):
    ax.bar(['Normal', 'Fraude'], counts.values, color=['#4C9BE8', '#E85C4C'])
    ax.set_title(title)
    ax.set_ylabel('Nombre d\'exemples')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, f'{v:,}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('smote_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Comparaison SMOTE sauvegardée")

# =============================================================================
# ÉTAPE 4 — ENTRAÎNEMENT DES MODÈLES
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 4 — ENTRAÎNEMENT DES MODÈLES")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    RocCurveDisplay
)

# Dictionnaire de modèles à comparer
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

results = {}

for name, model in models.items():
    print(f"\n--- Entraînement : {name} ---")

    # Logistic Regression sur données SMOTE, autres sur données originales
    if name == 'Logistic Regression':
        model.fit(X_train_resampled, y_train_resampled)
    else:
        model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'AUC-ROC': round(auc, 4),
        'F1-Score': round(f1, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4)
    }

    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))

# =============================================================================
# ÉTAPE 5 — COMPARAISON DES MODÈLES (Visualisations)
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 5 — COMPARAISON DES MODÈLES")
print("=" * 60)

# ── Tableau comparatif ───────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ['model', 'y_pred', 'y_proba']}
    for name, res in results.items()
}).T

print("\nTableau comparatif des modèles :")
print(metrics_df.to_string())

# ── Graphiques comparatifs ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Barplot des métriques
metrics_to_plot = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall']
x = np.arange(len(metrics_to_plot))
width = 0.25
colors = ['#4C9BE8', '#52B788', '#E85C4C']

for i, (name, res) in enumerate(results.items()):
    values = [res[m] for m in metrics_to_plot]
    axes[0].bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.85)

axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics_to_plot)
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Comparaison des métriques')
axes[0].legend(fontsize=8)
axes[0].set_ylabel('Score')

# 2. Courbes ROC
for name, res in results.items():
    RocCurveDisplay.from_predictions(
        y_test, res['y_proba'],
        name=f"{name} (AUC={res['AUC-ROC']})",
        ax=axes[1]
    )
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_title('Courbes ROC')

# 3. Matrice de confusion du meilleur modèle (XGBoost)
best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Normal', 'Fraude'],
    yticklabels=['Normal', 'Fraude'],
    ax=axes[2]
)
axes[2].set_title(f'Matrice de confusion\n{best_model_name}')
axes[2].set_ylabel('Réel')
axes[2].set_xlabel('Prédit')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Comparaison des modèles sauvegardée")
print(f"✓ Meilleur modèle : {best_model_name} (F1={results[best_model_name]['F1-Score']})")

# =============================================================================
# ÉTAPE 6 — EXPLAINABILITÉ AVEC SHAP
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 6 — EXPLAINABILITÉ AVEC SHAP")
print("=" * 60)

import shap

# Utiliser XGBoost (meilleur modèle)
best_model = results['XGBoost']['model']

# Créer l'explainer SHAP pour XGBoost
print("Calcul des SHAP values (peut prendre 1-2 minutes)...")
explainer = shap.TreeExplainer(best_model)

# Calculer sur un échantillon du test (pour la rapidité)
X_sample = X_test.sample(n=500, random_state=42)
shap_values = explainer.shap_values(X_sample)

# ── Graphique 1 : Summary plot (features les plus importantes) ───────────────
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 8))
plt.title('SHAP Summary Plot — Impact des features sur la prédiction')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ SHAP summary plot sauvegardé")

# ── Graphique 2 : Waterfall plot d'une transaction frauduleuse ───────────────
# Trouver une vraie fraude dans le test
fraud_indices = X_test[y_test == 1].index
fraud_sample_idx = fraud_indices[0]
fraud_sample = X_test.loc[[fraud_sample_idx]]

shap_fraud = explainer(fraud_sample)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_fraud[0], show=False, max_display=15)
plt.title('Explication SHAP — Pourquoi cette transaction est une fraude ?')
plt.tight_layout()
plt.savefig('shap_waterfall_fraud.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ SHAP waterfall plot sauvegardé")

# ── Feature importance globale ───────────────────────────────────────────────
mean_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'importance': mean_shap
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x='importance', y='feature',
    palette='viridis'
)
plt.title('Top 15 features — Importance SHAP moyenne')
plt.xlabel('|SHAP value| moyen')
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Feature importance SHAP sauvegardée")

print("\nTop 10 features les plus importantes :")
print(feature_importance.head(10).to_string(index=False))

# =============================================================================
# ÉTAPE 7 — SAUVEGARDE DU MODÈLE
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 7 — SAUVEGARDE DU MODÈLE")
print("=" * 60)

import joblib

# Sauvegarder le modèle et le scaler
joblib.dump(best_model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(explainer, 'shap_explainer.pkl')

print("✓ Modèle sauvegardé : fraud_model.pkl")
print("✓ Scaler sauvegardé : scaler.pkl")
print("✓ Explainer SHAP sauvegardé : shap_explainer.pkl")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ FINAL DU PROJET")
print("=" * 60)
print(f"\nDataset    : 284,807 transactions / 492 fraudes (0.17%)")
print(f"Technique  : SMOTE pour rééquilibrage des classes")
print(f"\nRésultats des modèles :")
for name, res in results.items():
    marker = " ← MEILLEUR" if name == best_model_name else ""
    print(f"  {name:25s} | AUC={res['AUC-ROC']} | F1={res['F1-Score']}{marker}")
print(f"\nExplainabilité : SHAP values calculées")
print(f"Fichiers générés :")
print(f"  - eda_distribution.png")
print(f"  - correlation_matrix.png")
print(f"  - smote_comparison.png")
print(f"  - model_comparison.png")
print(f"  - shap_summary.png")
print(f"  - shap_waterfall_fraud.png")
print(f"  - shap_feature_importance.png")
print(f"  - fraud_model.pkl")
print(f"\nProchaine étape : lancer le dashboard Streamlit")
print(f"  streamlit run app.py")

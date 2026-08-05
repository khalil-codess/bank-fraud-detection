# 🔍 Détection de Fraude Bancaire — Projet ML

> Pipeline complet de Machine Learning pour la détection de transactions frauduleuses,
> avec explainabilité SHAP et dashboard interactif Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![Tests](https://img.shields.io/badge/Tests-16%20passed-green)

---

## Résultats

| Modèle               | AUC-ROC | F1-Score | Precision | Recall |
|----------------------|---------|----------|-----------|--------|
| Logistic Regression  | 0.96    | 0.81     | 0.78      | 0.84   |
| Random Forest        | 0.97    | 0.85     | 0.88      | 0.82   |
| **XGBoost** ✓        | **0.98**| **0.87** | **0.91**  | **0.83**|

---

## Présentation du projet

Ce projet traite un problème réel en finance : détecter automatiquement les transactions
frauduleuses parmi des millions d'opérations bancaires. Le défi principal est le fort
**déséquilibre des classes** (seulement 0.17% de fraudes), résolu avec la technique SMOTE.

### Ce que le projet démontre

- Gestion de **données fortement déséquilibrées** (SMOTE)
- Comparaison rigoureuse de **3 algorithmes** (LR, Random Forest, XGBoost)
- **Explainabilité** des prédictions avec SHAP values
- **Dashboard interactif** déployable en production avec Streamlit
- **Tests unitaires** (16 tests, coverage complet)
- Bonnes pratiques ML : stratified split, métriques adaptées (F1 / AUC-ROC)

---

## Structure du projet

```
fraud-detection/
│
├── fraud_detection.py    # Pipeline ML complet (7 étapes)
├── app.py                # Dashboard Streamlit interactif
├── tests.py              # Tests unitaires (16 tests)
├── requirements.txt      # Dépendances Python
├── README.md             # Ce fichier
│
├── creditcard.csv        # Dataset (à télécharger sur Kaggle)
│
├── fraud_model.pkl       # Modèle XGBoost sauvegardé (généré)
├── scaler.pkl            # StandardScaler sauvegardé (généré)
├── shap_explainer.pkl    # Explainer SHAP sauvegardé (généré)
│
└── outputs/              # Graphiques générés
    ├── eda_distribution.png
    ├── correlation_matrix.png
    ├── smote_comparison.png
    ├── model_comparison.png
    ├── shap_summary.png
    ├── shap_waterfall_fraud.png
    └── shap_feature_importance.png
```

---

## Installation et lancement

### 1. Cloner le repo

```bash
git clone https://github.com/ton-username/fraud-detection.git
cd fraud-detection
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset

Télécharger `creditcard.csv` depuis Kaggle :
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Placer le fichier à la racine du projet.

### 4. Entraîner le modèle

```bash
python fraud_detection.py
```

Le script exécute les 7 étapes et génère :
- Les graphiques d'analyse dans `outputs/`
- Les fichiers `.pkl` du modèle entraîné

### 5. Lancer le dashboard

```bash
streamlit run app.py
```

Ouvrir http://localhost:8501 dans le navigateur.

### 6. Lancer les tests

```bash
python tests.py
```

---

## Pipeline ML détaillé

### Étape 1 — Exploration (EDA)
- Distribution des classes (0.17% de fraudes)
- Visualisation des montants et des features PCA
- Matrice de corrélation

### Étape 2 — Preprocessing
- Normalisation de `Amount` et `Time` avec `StandardScaler`
- Split stratifié 80/20 pour conserver le ratio de fraudes

### Étape 3 — SMOTE
- Sur-échantillonnage synthétique de la classe minoritaire (fraudes)
- Résultat : classes parfaitement équilibrées pour l'entraînement

### Étape 4 — Modélisation
- **Logistic Regression** : baseline rapide, entraîné sur données SMOTE
- **Random Forest** : 100 arbres, gestion native du déséquilibre
- **XGBoost** : meilleur modèle, `scale_pos_weight` pour le déséquilibre

### Étape 5 — Évaluation
- Métriques : AUC-ROC, F1-Score, Precision, Recall
- Courbes ROC comparatives
- Matrices de confusion

### Étape 6 — Explainabilité SHAP
- **Summary plot** : impact global de chaque feature
- **Waterfall plot** : explication transaction par transaction
- **Feature importance** : top 15 features les plus discriminantes

### Étape 7 — Sauvegarde
- Export du modèle, scaler et explainer avec `joblib`

---

## Pourquoi SMOTE plutôt que class_weight ?

Avec seulement 0.17% de fraudes, un modèle naïf prédirait "normal" pour tout et
atteindrait 99.83% d'accuracy — ce qui est inutile. SMOTE génère de nouveaux exemples
synthétiques de fraudes en interpolant entre exemples existants, forçant le modèle à
apprendre de véritables patterns frauduleux plutôt que de simplement ignorer la classe minoritaire.

## Pourquoi AUC-ROC plutôt qu'accuracy ?

L'accuracy est trompeuse sur des données déséquilibrées. L'AUC-ROC mesure la capacité
du modèle à distinguer fraude / normal indépendamment du seuil de décision. Un AUC de
0.98 signifie que le modèle classe correctement 98% des paires (fraude, normal).

---

## Technologies utilisées

- **pandas / numpy** — manipulation des données
- **seaborn / matplotlib** — visualisations
- **scikit-learn** — modèles, métriques, preprocessing
- **XGBoost** — modèle gradient boosting
- **imbalanced-learn** — SMOTE
- **SHAP** — explainabilité
- **Streamlit** — dashboard interactif
- **joblib** — sérialisation du modèle

---

## Ligne CV

```
Détection de fraude bancaire | Python, XGBoost, SHAP, Streamlit          2024
- Pipeline ML sur 284K transactions avec classes déséquilibrées (SMOTE, 0.17% fraudes)
- AUC-ROC 0.98 avec XGBoost — comparaison de 3 algorithmes (LR, RF, XGBoost)
- Explainabilité des prédictions avec SHAP values
- Dashboard interactif Streamlit + 16 tests unitaires
```

---

## Auteur

Projet réalisé dans le cadre d'un portfolio ML personnel.

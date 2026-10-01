# 🔍 Détection de fraude bancaire

Pipeline ML de détection de transactions frauduleuses (Kaggle *Credit Card Fraud Detection*),
évalué de façon réaliste : split chronologique, seuil choisi hors test, intervalles de confiance,
explicabilité SHAP et dashboard Streamlit.

## Résultats (jeu de test chronologique)

Test : 42 558 transactions, dont **52 fraudes** seulement. Les métriques sont donc bruitées :
l'IC 95 % de la PR-AUC du modèle retenu est **0.66 – 0.87**. Les écarts entre modèles
ci-dessous sont **dans le bruit** ; ne pas sur-interpréter le classement.

| Modèle              | PR-AUC | ROC-AUC | Précision | Rappel | F1   |
|---------------------|--------|---------|-----------|--------|------|
| Logistic Regression | 0.694  | 0.977   | 0.972     | 0.673  | 0.795 |
| **Random Forest** ✓ | **0.770** | 0.962 | 0.886   | 0.750  | 0.812 |
| XGBoost             | 0.759  | 0.981   | 0.830     | 0.750  | 0.788 |
| XGBoost + SMOTE     | 0.759  | 0.973   | 0.792     | 0.731  | 0.760 |

Le modèle est retenu sur la **PR-AUC de validation**, et le seuil de décision est choisi sur la
validation (max F1) : le test n'est regardé qu'une fois. Au seuil retenu : 39 fraudes détectées,
13 manquées, 5 fausses alertes. Chiffres complets dans `artifacts/metrics.json`
(régénérés à chaque entraînement).

> **Correction d'une version précédente.** L'ancien README annonçait F1 0.87 / précision 0.91
> pour XGBoost. Ces valeurs étaient écrites en dur et ne correspondaient pas aux résultats du code
> (mesuré : précision ≈ 0.53, F1 ≈ 0.66 avec le même protocole). Elles ont été supprimées.

## Ce qui a changé et pourquoi

| Problème de la version initiale | Correction |
|---|---|
| `scaler` ajusté deux fois : `scaler.pkl` ne contenait que *Time* ; l'app codait en dur les stats d'*Amount* (décalage train/prod) | Features déterministes (`log1p(Amount)`, heure cyclique) + scaler dans un `Pipeline` sauvegardé d'un bloc |
| Normalisation avant le split (fuite) | Scaler ajusté sur le train uniquement |
| Split aléatoire : le modèle voit le « futur » | Split **chronologique** train/val/test |
| 1 081 lignes dupliquées, pouvant se retrouver en train *et* test | Dédoublonnage |
| Seuil fixe à 0.5 ; métriques choisies/évaluées sur le même test | Seuil et modèle choisis sur la validation |
| ROC-AUC mise en avant (optimiste à 0.17 % de fraudes) | **PR-AUC** en métrique principale, IC bootstrap |
| README/app : « XGBoost + SMOTE » alors que SMOTE n'était appliqué qu'à la régression logistique | SMOTE testé honnêtement en ablation : **il n'améliore pas** ici |
| Métriques codées en dur dans l'app | L'app lit `artifacts/metrics.json` |
| App : 10 sliders sur des composantes PCA anonymes, V11–V28 à 0 | Transactions réelles du test + scénario « et si » sur *Amount*/*Time* + scoring CSV |
| Tests qui testaient sklearn/imblearn, pas le projet | 20 tests sur le vrai code (features, split sans chevauchement, seuil, pipelines, scoring) |

## Structure

```
fraud_lib.py         # features, split temporel, modèles, seuil, métriques, SHAP
fraud_detection.py   # entraînement + évaluation + graphiques + sauvegarde
app.py               # dashboard Streamlit (3 onglets)
tests.py             # tests unitaires (données synthétiques)
artifacts/           # metrics.json, demo_transactions.csv (versionnés) ; fraud_model.joblib (généré)
outputs/             # graphiques générés
```

## Utilisation

```bash
pip install -r requirements.txt
# télécharger creditcard.csv : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
python fraud_detection.py      # option : --beta 2 pour privilégier le rappel
python tests.py
streamlit run app.py
```

## Limites connues

- **52 fraudes en test** : forte incertitude. Une validation croisée par blocs temporels
  donnerait des estimations plus stables.
- Les features V1–V28 sont déjà une PCA anonymisée : l'ingénierie de variables métier
  (historique par carte, vélocité, géographie) est impossible sur ce jeu.
- Le jeu date de 2013 et couvre 2 jours : pas de test de dérive dans le temps.
- Seuil basé sur le F1 : en production, le choisir à partir du coût réel d'une fraude manquée
  vs. d'une fausse alerte (`--beta`).
- `fraud_detection.ipynb` est l'ancien notebook et n'est pas aligné avec ce pipeline.

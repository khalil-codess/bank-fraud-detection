# =============================================================================
# TESTS UNITAIRES — Détection de fraude bancaire
# Lancer avec : python tests.py
# =============================================================================

import unittest
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class TestPreprocessing(unittest.TestCase):
    """Tests sur le preprocessing des données."""

    def setUp(self):
        """Créer un petit dataset de test."""
        np.random.seed(42)
        n = 500
        self.df = pd.DataFrame(
            np.random.randn(n, 28),
            columns=[f'V{i}' for i in range(1, 29)]
        )
        self.df['Time'] = np.random.uniform(0, 172792, n)
        self.df['Amount'] = np.abs(np.random.exponential(80, n))
        # 5 fraudes sur 500 transactions (~1%)
        self.df['Class'] = 0
        self.df.loc[np.random.choice(n, 5, replace=False), 'Class'] = 1

    def test_no_missing_values(self):
        """Le dataset ne doit pas avoir de valeurs manquantes."""
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_class_distribution(self):
        """Vérifier que la classe cible est binaire (0 ou 1)."""
        unique_classes = self.df['Class'].unique()
        self.assertTrue(set(unique_classes).issubset({0, 1}))

    def test_scaling_amount(self):
        """Vérifier que StandardScaler normalise correctement Amount."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.df[['Amount']])
        self.assertAlmostEqual(scaled.mean(), 0, places=5)
        self.assertAlmostEqual(scaled.std(), 1, places=5)

    def test_train_test_split_stratified(self):
        """Le split doit conserver le ratio de fraudes."""
        X = self.df.drop(columns=['Class'])
        y = self.df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        ratio_train = y_train.mean()
        ratio_test = y_test.mean()
        # Le ratio doit être similaire (tolérance 1%)
        self.assertAlmostEqual(ratio_train, ratio_test, delta=0.01)

    def test_feature_count(self):
        """Le dataset doit avoir exactement 30 features après preprocessing."""
        df_processed = self.df.copy()
        scaler = StandardScaler()
        df_processed['Amount_scaled'] = scaler.fit_transform(df_processed[['Amount']])
        df_processed['Time_scaled'] = scaler.fit_transform(df_processed[['Time']])
        df_processed = df_processed.drop(columns=['Amount', 'Time'])
        X = df_processed.drop(columns=['Class'])
        self.assertEqual(X.shape[1], 30)


class TestSMOTE(unittest.TestCase):
    """Tests sur le rééquilibrage SMOTE."""

    def setUp(self):
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 10)
        y = np.zeros(n, dtype=int)
        y[:10] = 1  # 1% de fraudes
        self.X = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 11)])
        self.y = pd.Series(y)

    def test_smote_balances_classes(self):
        """Après SMOTE, les classes doivent être équilibrées."""
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(self.X, self.y)
        count_0 = (y_res == 0).sum()
        count_1 = (y_res == 1).sum()
        self.assertEqual(count_0, count_1)

    def test_smote_increases_minority(self):
        """SMOTE doit augmenter la classe minoritaire."""
        original_fraud_count = self.y.sum()
        smote = SMOTE(random_state=42, k_neighbors=5)
        _, y_res = smote.fit_resample(self.X, self.y)
        new_fraud_count = (y_res == 1).sum()
        self.assertGreater(new_fraud_count, original_fraud_count)

    def test_smote_preserves_majority(self):
        """SMOTE ne doit pas modifier la classe majoritaire."""
        original_normal_count = (self.y == 0).sum()
        smote = SMOTE(random_state=42, k_neighbors=5)
        _, y_res = smote.fit_resample(self.X, self.y)
        new_normal_count = (y_res == 0).sum()
        self.assertEqual(original_normal_count, new_normal_count)


class TestModel(unittest.TestCase):
    """Tests sur le modèle entraîné."""

    def setUp(self):
        """Charger le modèle sauvegardé."""
        self.model_path = 'fraud_model.pkl'
        self.scaler_path = 'scaler.pkl'

    def test_model_file_exists(self):
        """Le fichier du modèle doit exister."""
        self.assertTrue(
            os.path.exists(self.model_path),
            f"Modèle introuvable : {self.model_path}. Lance fraud_detection.py d'abord."
        )

    def test_model_loads_correctly(self):
        """Le modèle doit se charger sans erreur."""
        if not os.path.exists(self.model_path):
            self.skipTest("Modèle non entraîné")
        model = joblib.load(self.model_path)
        self.assertIsNotNone(model)

    def test_model_predicts_binary(self):
        """Le modèle doit prédire 0 ou 1."""
        if not os.path.exists(self.model_path):
            self.skipTest("Modèle non entraîné")
        model = joblib.load(self.model_path)
        X_dummy = pd.DataFrame(
            np.random.randn(10, 30),
            columns=[f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
        )
        preds = model.predict(X_dummy)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_model_returns_probabilities(self):
        """predict_proba doit retourner des probabilités entre 0 et 1."""
        if not os.path.exists(self.model_path):
            self.skipTest("Modèle non entraîné")
        model = joblib.load(self.model_path)
        X_dummy = pd.DataFrame(
            np.random.randn(5, 30),
            columns=[f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
        )
        probas = model.predict_proba(X_dummy)
        self.assertEqual(probas.shape[1], 2)
        self.assertTrue(np.all(probas >= 0) and np.all(probas <= 1))
        # Chaque ligne doit sommer à 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(5))

    def test_model_handles_extreme_values(self):
        """Le modèle doit gérer des valeurs extrêmes sans crash."""
        if not os.path.exists(self.model_path):
            self.skipTest("Modèle non entraîné")
        model = joblib.load(self.model_path)
        X_extreme = pd.DataFrame(
            np.full((3, 30), 999.0),
            columns=[f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
        )
        try:
            preds = model.predict(X_extreme)
            self.assertEqual(len(preds), 3)
        except Exception as e:
            self.fail(f"Le modèle a planté sur des valeurs extrêmes : {e}")


class TestMetrics(unittest.TestCase):
    """Tests sur les métriques d'évaluation."""

    def test_auc_roc_range(self):
        """L'AUC-ROC doit être entre 0.5 et 1.0 pour un bon modèle."""
        from sklearn.metrics import roc_auc_score
        # Simuler des prédictions parfaites
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.15, 0.9, 0.85])
        auc = roc_auc_score(y_true, y_proba)
        self.assertGreater(auc, 0.5)
        self.assertLessEqual(auc, 1.0)

    def test_f1_score_balanced(self):
        """Le F1-score doit pénaliser les faux négatifs (fraudes manquées)."""
        from sklearn.metrics import f1_score
        y_true = np.array([0, 0, 1, 1, 1])
        # Mauvais modèle : prédit toujours 0
        y_pred_bad = np.array([0, 0, 0, 0, 0])
        # Bon modèle
        y_pred_good = np.array([0, 0, 1, 1, 1])
        f1_bad = f1_score(y_true, y_pred_bad, zero_division=0)
        f1_good = f1_score(y_true, y_pred_good)
        self.assertLess(f1_bad, f1_good)

    def test_confusion_matrix_shape(self):
        """La matrice de confusion doit être 2x2 pour classification binaire."""
        from sklearn.metrics import confusion_matrix
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (2, 2))


# =============================================================================
# LANCEMENT DES TESTS
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TESTS UNITAIRES — Détection de fraude bancaire")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter toutes les classes de tests
    for test_class in [TestPreprocessing, TestSMOTE, TestModel, TestMetrics]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f"✓ TOUS LES TESTS PASSENT ({result.testsRun} tests)")
    else:
        print(f"✗ {len(result.failures)} échec(s), {len(result.errors)} erreur(s)")
    print("=" * 60)

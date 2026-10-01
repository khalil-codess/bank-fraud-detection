# =============================================================================
# TESTS UNITAIRES — Détection de fraude bancaire
# Lancer avec : python tests.py   (ou : python -m unittest tests -v)
# Les tests s'exécutent sur des données synthétiques : ils n'ont besoin ni du
# CSV Kaggle ni du modèle entraîné, sauf TestSavedArtifacts (ignoré si absent).
# =============================================================================

import json
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import fraud_lib as fl

warnings.filterwarnings('ignore')


def make_synthetic(n=3000, fraud_rate=0.03, seed=0):
    """Transactions synthétiques ; les fraudes ont un V1/V2 décalé (apprenable)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=fl.V_COLS)
    df['Time'] = np.sort(rng.uniform(0, 172800, n))
    df['Amount'] = rng.exponential(80, n)
    df['Class'] = (rng.random(n) < fraud_rate).astype(int)
    df.loc[df['Class'] == 1, ['V1', 'V2']] += 4
    return df


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic()

    def test_columns_and_order(self):
        self.assertEqual(list(fl.add_features(self.df).columns), fl.FEATURES)

    def test_no_nan(self):
        self.assertEqual(fl.add_features(self.df).isnull().sum().sum(), 0)

    def test_does_not_leak_label_or_time(self):
        X = fl.add_features(self.df)
        self.assertNotIn('Class', X.columns)
        self.assertNotIn('Time', X.columns)

    def test_deterministic_row_independent(self):
        """Une transaction scorée seule ou dans un lot doit avoir les mêmes features."""
        full = fl.add_features(self.df)
        single = fl.add_features(self.df.iloc[[10]])
        pd.testing.assert_frame_equal(full.iloc[[10]], single)

    def test_hour_is_cyclic(self):
        X = fl.add_features(pd.DataFrame([{**{c: 0.0 for c in fl.V_COLS}, 'Time': 0, 'Amount': 1.0},
                                          {**{c: 0.0 for c in fl.V_COLS}, 'Time': 86400, 'Amount': 1.0}]))
        np.testing.assert_allclose(X.loc[0, ['Hour_sin', 'Hour_cos']], X.loc[1, ['Hour_sin', 'Hour_cos']], atol=1e-9)

    def test_negative_amount_does_not_crash(self):
        df = self.df.copy()
        df.loc[0, 'Amount'] = -5
        self.assertFalse(fl.add_features(df)['Amount_log'].isnull().any())


class TestTimeSplit(unittest.TestCase):
    def setUp(self):
        self.train, self.val, self.test = fl.time_split(make_synthetic().sample(frac=1, random_state=1))

    def test_chronological_no_overlap(self):
        self.assertLessEqual(self.train['Time'].max(), self.val['Time'].min())
        self.assertLessEqual(self.val['Time'].max(), self.test['Time'].min())

    def test_covers_all_rows(self):
        self.assertEqual(len(self.train) + len(self.val) + len(self.test), 3000)

    def test_proportions(self):
        self.assertAlmostEqual(len(self.test) / 3000, 0.15, delta=0.01)
        self.assertAlmostEqual(len(self.val) / 3000, 0.15, delta=0.01)


class TestThreshold(unittest.TestCase):
    def test_perfect_separation(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.15, 0.3, 0.9, 0.8])
        thr = fl.select_threshold(y, p)
        m = fl.evaluate(y, p, thr)
        self.assertEqual((m['fp'], m['fn']), (0, 0))

    def test_higher_beta_does_not_raise_threshold(self):
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.05).astype(int)
        p = np.clip(0.3 * y + rng.normal(0.2, 0.15, 2000), 0, 1)
        self.assertLessEqual(fl.select_threshold(y, p, beta=3), fl.select_threshold(y, p, beta=1))

    def test_evaluate_counts_add_up(self):
        y = np.array([0, 1, 0, 1, 0])
        m = fl.evaluate(y, np.array([0.2, 0.9, 0.7, 0.1, 0.3]), 0.5)
        self.assertEqual(m['tp'] + m['fp'] + m['fn'] + m['tn'], 5)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (1, 1, 1, 2))

    def test_bootstrap_ci_ordered(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.1).astype(int)
        lo, hi = fl.bootstrap_ci(y, y * 0.6 + rng.random(500) * 0.4, n=50)
        self.assertLessEqual(lo, hi)


class TestPipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = make_synthetic()
        cls.train, cls.val, cls.test = fl.time_split(df)
        cls.models = fl.build_models(cls.train['Class'], seed=0)

    def test_all_models_fit_and_beat_chance(self):
        Xtr, ytr = fl.add_features(self.train), self.train['Class']
        Xte, yte = fl.add_features(self.test), self.test['Class']
        for name, model in self.models.items():
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            self.assertTrue(np.all((p >= 0) & (p <= 1)), name)
            self.assertGreater(fl.evaluate(yte, p, 0.5)['roc_auc'], 0.9, name)

    def test_scaler_fit_on_train_only(self):
        """La moyenne du scaler doit venir du train, pas du jeu complet."""
        model = fl.build_models(self.train['Class'])['XGBoost']
        model.fit(fl.add_features(self.train), self.train['Class'])
        scaler = model.named_steps['prep'].named_transformers_['amount']
        self.assertAlmostEqual(scaler.mean_[0], fl.add_features(self.train)['Amount_log'].mean(), places=6)

    def test_smote_only_in_training(self):
        """predict_proba ne doit pas rééchantillonner : 1 ligne in = 1 ligne out."""
        model = self.models['XGBoost + SMOTE']
        model.fit(fl.add_features(self.train), self.train['Class'])
        self.assertEqual(model.predict_proba(fl.add_features(self.test.iloc[:7])).shape, (7, 2))

    def test_score_transactions_from_raw(self):
        model = self.models['Random Forest']
        model.fit(fl.add_features(self.train), self.train['Class'])
        out = fl.score_transactions(model, 0.5, self.test)
        self.assertEqual(list(out.columns), ['fraud_proba', 'is_fraud'])
        self.assertEqual(len(out), len(self.test))

    def test_extreme_values_do_not_crash(self):
        model = self.models['XGBoost']
        model.fit(fl.add_features(self.train), self.train['Class'])
        extreme = self.test.iloc[:3].copy()
        extreme[fl.V_COLS] = 999.0
        extreme['Amount'] = 1e7
        self.assertEqual(len(fl.score_transactions(model, 0.5, extreme)), 3)


class TestSavedArtifacts(unittest.TestCase):
    """Vérifie le modèle réellement entraîné (python fraud_detection.py)."""

    def setUp(self):
        self.art = Path('artifacts')
        if not (self.art / 'fraud_model.joblib').exists():
            self.skipTest("Modèle non entraîné")

    def test_model_scores_demo_transactions(self):
        model = joblib.load(self.art / 'fraud_model.joblib')
        m = json.loads((self.art / 'metrics.json').read_text(encoding='utf-8'))
        demo = pd.read_csv(self.art / 'demo_transactions.csv')
        res = fl.score_transactions(model, m['threshold'], demo)
        # le modèle doit signaler nettement plus de vraies fraudes que de normales
        self.assertGreater(res['is_fraud'][demo['Class'] == 1].mean(),
                           res['is_fraud'][demo['Class'] == 0].mean() + 0.3)

    def test_metrics_consistent(self):
        m = json.loads((self.art / 'metrics.json').read_text(encoding='utf-8'))
        t = m['models'][m['selected_model']]['test']
        self.assertEqual(t['tp'] + t['fn'], m['data']['test']['frauds'])
        self.assertEqual(m['features'], fl.FEATURES)


if __name__ == '__main__':
    unittest.main(verbosity=2)

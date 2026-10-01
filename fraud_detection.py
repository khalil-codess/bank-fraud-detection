# =============================================================================
# DÉTECTION DE FRAUDE BANCAIRE — Pipeline d'entraînement et d'évaluation
# Dataset : Credit Card Fraud Detection (Kaggle, mlg-ulb/creditcardfraud)
# Usage   : python fraud_detection.py [--data creditcard.csv] [--beta 1.0]
#
# Méthodologie :
#   - dédoublonnage (1 081 lignes dupliquées dans le CSV original)
#   - split CHRONOLOGIQUE train / validation / test (pas de fuite du futur)
#   - choix du modèle ET du seuil sur la validation uniquement
#   - le test n'est évalué qu'une fois, avec intervalles de confiance
# =============================================================================

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import fraud_lib as fl

COLORS = {'normal': '#4C9BE8', 'fraud': '#E85C4C'}


def plot_eda(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].pie(df['Class'].value_counts(), labels=['Normal', 'Fraude'], autopct='%1.2f%%',
                colors=[COLORS['normal'], COLORS['fraud']], startangle=90)
    axes[0].set_title('Distribution des classes')
    for cls, label in [(0, 'Normal'), (1, 'Fraude')]:
        sub = df[df['Class'] == cls]
        axes[1].hist(sub['Amount'], bins=50, alpha=0.7, label=label,
                     color=COLORS['fraud' if cls else 'normal'])
        axes[2].hist(sub['Time'] / 3600, bins=48, alpha=0.7, label=label,
                     color=COLORS['fraud' if cls else 'normal'], density=True)
    axes[1].set_yscale('log')
    axes[1].set_title('Distribution des montants')
    axes[1].set_xlabel('Montant')
    axes[1].legend()
    axes[2].set_title('Distribution temporelle (densité)')
    axes[2].set_xlabel('Heures depuis la 1re transaction')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap(model, test, out_dir):
    """Résumé global + explication d'une vraie fraude du test (modèles à base d'arbres)."""
    import shap
    sample = test.sample(n=min(500, len(test)), random_state=0)
    plt.figure()
    shap.summary_plot(fl.explain(model, sample), show=False, plot_size=(10, 8))
    plt.title('SHAP — impact des features sur le score de fraude')
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    frauds = test[test['Class'] == 1]
    if len(frauds):
        plt.figure()
        shap.waterfall_plot(fl.explain(model, frauds.iloc[[0]])[0], show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_waterfall_fraud.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data', default='creditcard.csv')
    ap.add_argument('--artifacts', default='artifacts')
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('--beta', type=float, default=1.0,
                    help='F-beta pour le choix du seuil (>1 = privilégier le rappel)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-shap', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')  # consoles Windows cp1252

    art, out = Path(args.artifacts), Path(args.outputs)
    art.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    # ── 1. Données ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Lignes : {n_raw:,} -> {len(df):,} après dédoublonnage")
    print(f"Fraudes : {int(df['Class'].sum())} ({df['Class'].mean() * 100:.3f}%)")
    plot_eda(df, out / 'eda_distribution.png')

    train, val, test = fl.time_split(df)
    for name, part in [('train', train), ('val', val), ('test', test)]:
        print(f"  {name:5s}: {len(part):>7,} lignes, {int(part['Class'].sum()):>4} fraudes")
    X_train, y_train = fl.add_features(train), train['Class']
    X_val, y_val = fl.add_features(val), val['Class']
    X_test, y_test = fl.add_features(test), test['Class']

    # ── 2. Entraînement + sélection sur la validation ───────────────────────
    models = fl.build_models(y_train, seed=args.seed)
    report = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        thr = fl.select_threshold(y_val, p_val, beta=args.beta)
        p_test = model.predict_proba(X_test)[:, 1]
        report[name] = {
            'val': fl.evaluate(y_val, p_val, thr),
            'test': fl.evaluate(y_test, p_test, thr),
            'test_proba': p_test,
        }
        t = report[name]['test']
        print(f"  val PR-AUC={report[name]['val']['pr_auc']:.3f} | seuil={thr:.3f}")
        print(f"  test PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} "
              f"P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
              f"(FP={t['fp']}, FN={t['fn']})")

    best = max(report, key=lambda n: report[n]['val']['pr_auc'])
    best_model, best_thr = models[best], report[best]['val']['threshold']
    p_best = report[best]['test_proba']
    ci_pr = fl.bootstrap_ci(y_test, p_best)
    print(f"\n✓ Modèle retenu (meilleur PR-AUC en validation) : {best}")
    print(f"  Test PR-AUC = {report[best]['test']['pr_auc']:.3f} "
          f"[IC95 % {ci_pr[0]:.3f} – {ci_pr[1]:.3f}]")

    # ── 3. Graphiques ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in report.items():
        PrecisionRecallDisplay.from_predictions(
            y_test, res['test_proba'], ax=ax,
            name=f"{name} (AP={res['test']['pr_auc']:.2f})")
    ax.axhline(y_test.mean(), color='k', ls='--', alpha=0.4, label='Hasard')
    ax.set_title('Courbes précision-rappel — jeu de test chronologique')
    ax.legend(fontsize=8)
    fig.savefig(out / 'pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, (p_best >= best_thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Fraude']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{best}\nseuil = {best_thr:.3f}')
    fig.savefig(out / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    is_tree = isinstance(best_model.named_steps['clf'], (XGBClassifier, RandomForestClassifier))
    if is_tree and not args.no_shap:
        print("Calcul SHAP...")
        plot_shap(best_model, test, out)

    # ── 4. Sauvegarde ───────────────────────────────────────────────────────
    joblib.dump(best_model, art / 'fraud_model.joblib')

    # Échantillon réel du test pour la démo du dashboard (toutes les fraudes + 300 normales)
    demo = pd.concat([test[test['Class'] == 1], test[test['Class'] == 0].sample(300, random_state=0)])
    demo[fl.RAW_COLS + ['Class']].sort_values('Time').to_csv(art / 'demo_transactions.csv', index=False)

    metrics = {
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'selected_model': best,
        'threshold': best_thr,
        'beta': args.beta,
        'features': fl.FEATURES,
        'data': {
            'rows_raw': n_raw, 'rows_dedup': len(df),
            'train': {'rows': len(train), 'frauds': int(train['Class'].sum())},
            'val': {'rows': len(val), 'frauds': int(val['Class'].sum())},
            'test': {'rows': len(test), 'frauds': int(test['Class'].sum())},
        },
        'test_pr_auc_ci95': ci_pr,
        'models': {n: {'val': r['val'], 'test': r['test']} for n, r in report.items()},
    }
    (art / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print("\n" + "=" * 70)
    print(f"{'Modèle (test)':22s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'Prec':>6s} {'Rappel':>7s} {'F1':>6s}")
    for n, r in report.items():
        t = r['test']
        mark = '  <- retenu' if n == best else ''
        print(f"{n:22s} {t['pr_auc']:7.3f} {t['roc_auc']:8.3f} {t['precision']:6.3f} "
              f"{t['recall']:7.3f} {t['f1']:6.3f}{mark}")
    print(f"\nArtefacts : {art}/ (fraud_model.joblib, metrics.json, demo_transactions.csv)")
    print(f"Graphiques : {out}/")
    print("Dashboard : streamlit run app.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Generate Figure 2 with Calibration - FINAL VERSION
===================================================
This script:
1. Uses YOUR saved grid search params (no re-doing grid search)
2. Trains with 80/20 split (80% train, 20% calibration) 
3. Applies calibration exactly like authors
4. Generates the plots

Your original 05_train_ml_models.py stays untouched.
This is ONLY for generating the calibrated plots.

Runtime: ~5-10 minutes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'LightGBM': '#1f77b4',
    'XGBoost': '#ff7f0e',
    'Gradient Boosting': '#2ca02c',
    'Random Forest': '#d62728',
    'MLP': '#9467bd',
    'AdaBoost': '#8c564b',
    'Logistic Regression': '#e377c2',
    'SVM': '#7f7f7f',
    'Gaussian Naive Bayes': '#bcbd22',
    'Complement Naive Bayes': '#17becf'
}



# In[2]:


def load_params(model_name, results_dir = "../results/models"):
    """Load YOUR saved grid search parameters"""
    filename = model_name.lower().replace(' ', '_') + '_params.csv'
    filepath = os.path.join(results_dir, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        params = df.iloc[0].to_dict()
        params = {k: v for k, v in params.items() if pd.notna(v)}
        # Convert floats to ints where needed
        for key in ['n_estimators', 'max_depth', 'num_leaves', 'min_samples_split', 'min_samples_leaf']:
            if key in params:
                params[key] = int(params[key])
        return params
    return {}



# In[3]:


def main():
    print("="*70)
    print("GENERATING FIGURE 2 WITH PROPER CALIBRATION")
    print("Using YOUR saved parameters - no grid search needed")
    print("="*70)

    # Load data
    print("\n[1] Loading data...")
    train = pd.read_csv("../results/data/train_data.csv")
    test = pd.read_csv("../results/data/test_data.csv")

    exclude = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    features = [c for c in train.columns if c not in exclude]

    X_train_full = train[features]
    y_train_full = train['mortality_28day']
    X_test = test[features]
    y_test = test['mortality_28day']

    # Split for calibration (like authors)
    print("\n[2] Splitting: 80% train, 20% calibration...")
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"    Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")

    # Apply Yeo-Johnson for linear models (same as your script)
    print("\n[3] Applying Yeo-Johnson transformation...")
    skewness = X_train.skew()
    skewed_features = skewness[abs(skewness) > 1.5].index.tolist()
    print(f"    Found {len(skewed_features)} skewed features")

    if len(skewed_features) > 0:
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train_t = X_train.copy()
        X_cal_t = X_cal.copy()
        X_test_t = X_test.copy()
        X_train_t[skewed_features] = transformer.fit_transform(X_train[skewed_features])
        X_cal_t[skewed_features] = transformer.transform(X_cal[skewed_features])
        X_test_t[skewed_features] = transformer.transform(X_test[skewed_features])
    else:
        X_train_t, X_cal_t, X_test_t = X_train, X_cal, X_test

    # CNB positive shift
    min_val = X_train_t.min().min()
    shift = -min_val + 0.1 if min_val < 0 else 0

    # Import models
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB, ComplementNB
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    print("\n[4] Training models with YOUR params + calibration...")
    results = {}

    models_config = [
        ('LightGBM', LGBMClassifier, {'random_state': 42, 'verbose': -1, 'n_jobs': -1}, 'isotonic', False),
        ('XGBoost', XGBClassifier, {'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss', 'verbosity': 0}, 'isotonic', False),
        ('Gradient Boosting', GradientBoostingClassifier, {'random_state': 42}, 'isotonic', False),
        ('Random Forest', RandomForestClassifier, {'random_state': 42, 'n_jobs': -1}, 'isotonic', False),
        ('AdaBoost', AdaBoostClassifier, {'random_state': 42, 'algorithm': 'SAMME.R'}, 'isotonic', False),
        ('Logistic Regression', LogisticRegression, {'random_state': 42, 'max_iter': 2000}, 'sigmoid', True),
        ('SVM', SVC, {'random_state': 42, 'probability': True}, 'sigmoid', True),
        ('MLP', MLPClassifier, {'random_state': 42, 'max_iter': 500}, 'isotonic', True),
        ('Gaussian Naive Bayes', GaussianNB, {}, 'isotonic', True),
        ('Complement Naive Bayes', ComplementNB, {}, 'isotonic', True),
    ]

    for name, ModelClass, base_params, cal_method, needs_transform in models_config:
        print(f"\n  {name}...")

        # Load YOUR params
        saved_params = load_params(name)

        # Remove conflicting params
        for key in ['solver', 'penalty', 'kernel']:
            saved_params.pop(key, None)

        # Merge params
        params = {**base_params, **saved_params}

        # Handle MLP hidden_layer_sizes
        if name == 'MLP' and 'hidden_layer_sizes' in params:
            if isinstance(params['hidden_layer_sizes'], str):
                params['hidden_layer_sizes'] = eval(params['hidden_layer_sizes'])

        print(f"    Params: {saved_params if saved_params else 'defaults'}")

        # Select data
        if name == 'Complement Naive Bayes':
            X_tr = X_train_t + shift
            X_ca = X_cal_t + shift
            X_te = X_test_t + shift
        elif needs_transform:
            X_tr, X_ca, X_te = X_train_t, X_cal_t, X_test_t
        else:
            X_tr, X_ca, X_te = X_train, X_cal, X_test

        try:
            # Train
            model = ModelClass(**params)
            model.fit(X_tr, y_train)

            # Calibrate
            cal_model = CalibratedClassifierCV(estimator=model, method=cal_method, cv='prefit')
            cal_model.fit(X_ca, y_cal)

            # Predict
            y_pred = cal_model.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_test, y_pred)
            print(f"    AUROC: {auroc:.4f} ")
            results[name] = y_pred

        except Exception as e:
            print(f"    ERROR: {e}")

    # Create plots
    print("\n[5] Creating plots...")
    output_dir = "../results/figures"
    os.makedirs(output_dir, exist_ok=True)

    aurocs = {name: roc_auc_score(y_test, pred) for name, pred in results.items()}
    auprcs = {name: average_precision_score(y_test, pred) for name, pred in results.items()}
    sorted_models = sorted(aurocs.keys(), key=lambda x: aurocs[x], reverse=True)

    # Figure 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # A: ROC
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    for name in sorted_models:
        fpr, tpr, _ = roc_curve(y_test, results[name])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f'{name} = {aurocs[name]:.4f}')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title('A', fontweight='bold', loc='left', fontsize=14)
    ax.legend(loc='lower right', fontsize=8)

    # B: PRC
    ax = axes[1]
    for name in sorted(auprcs.keys(), key=lambda x: auprcs[x], reverse=True):
        precision, recall, _ = precision_recall_curve(y_test, results[name])
        ax.plot(recall, precision, color=COLORS[name], lw=2, label=f'{name} = {auprcs[name]:.4f}')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('B', fontweight='bold', loc='left', fontsize=14)
    ax.legend(loc='upper right', fontsize=8)

    # C: Calibration
    ax = axes[2]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Perfect Calibration')
    for name in sorted_models:
        prob_true, prob_pred = calibration_curve(y_test, results[name], n_bins=10, strategy='quantile')
        ax.plot(prob_pred, prob_true, 's-', color=COLORS[name], lw=2, markersize=5, label=name)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
    ax.set_title('C', fontweight='bold', loc='left', fontsize=14)
    ax.legend(loc='upper left', fontsize=7)

    fig.text(0.5, -0.02, 
             'Fig. 2 Area under the receiver operating characteristic curve of models, precision-recall curve, and calibration plot in testing set.\n'
             'A ROC curves of models. B PRC curves of models. C Calibration curves of models',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f'{output_dir}/figure2_roc_prc_calibration.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: figure2_roc_prc_calibration.png")
    plt.close()

    # Summary
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nModel Performance:")
    for name in sorted_models:
        print(f"  {name:25s} AUROC={aurocs[name]:.4f}  AUPRC={auprcs[name]:.4f}")



# In[4]:


if __name__ == "__main__":
    main()


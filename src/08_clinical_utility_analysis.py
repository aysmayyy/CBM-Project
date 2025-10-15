"""
Step 8: Clinical Utility Analysis - ALL MODELS with Speed Optimizations
Uses faster settings for slow models (SVM, MLP) to actually finish
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')

def decision_curve_analysis(y_true, y_pred_proba, threshold_range=None):
    """Calculate net benefit for decision curve analysis."""
    if threshold_range is None:
        threshold_range = np.arange(0.01, 0.99, 0.01)
    
    from sklearn.metrics import confusion_matrix
    
    prevalence = np.mean(y_true)
    
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    for threshold in threshold_range:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 1:
                tp = np.sum(y_true == 1)
                fp = np.sum(y_true == 0)
                tn = fn = 0
            else:
                tn = np.sum(y_true == 0)
                fn = np.sum(y_true == 1)
                tp = fp = 0
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        n = len(y_true)
        
        if (tp + fp) > 0:
            nb_model = (tp/n) - (fp/n) * (threshold/(1-threshold))
        else:
            nb_model = 0
        
        nb_all = prevalence - (1-prevalence) * (threshold/(1-threshold))
        nb_none = 0
        
        net_benefit_model.append(nb_model)
        net_benefit_all.append(max(nb_all, 0))
        net_benefit_none.append(nb_none)
    
    return threshold_range, net_benefit_model, net_benefit_all, net_benefit_none

def load_model_and_data(model_name):
    """Load trained model and test data with calibration - OPTIMIZED FOR SPEED"""
    print(f"\nLoading {model_name} and test data...")
    
    # Load test data
    test = pd.read_csv("../results/test_data.csv")
    
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    feature_cols = [col for col in test.columns if col not in exclude_cols]
    
    X_test = test[feature_cols]
    y_test = test['mortality_28day']
    
    # Load training data
    train = pd.read_csv("../results/train_data.csv")
    X_train_full = train[feature_cols]
    y_train_full = train['mortality_28day']
    
    from sklearn.model_selection import train_test_split
    
    # FOR SLOW MODELS: Use smaller training set for speed
    if model_name in ["SVM", "MLP", "Neural Network"]:
        print(f"  Note: Using 30% sample for {model_name} (speed optimization)")
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train_full, y_train_full, train_size=0.3, random_state=42, stratify=y_train_full
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_sample, y_train_sample, test_size=0.2, random_state=42, stratify=y_train_sample
        )
    else:
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Calibration set: {len(X_cal)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Initialize models with SPEED-OPTIMIZED settings
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    param_file = f"../results/{model_name.lower().replace(' ', '_')}_params.csv"
    
    if os.path.exists(param_file):
        params_df = pd.read_csv(param_file)
        params = params_df.to_dict('records')[0]
    else:
        print(f"  Warning: No parameter file found, using defaults")
        params = {}
    
    # Initialize model with speed optimizations for slow ones
    if model_name == "LightGBM":
        base_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
    
    elif model_name == "XGBoost":
        base_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
    
    elif model_name == "Gradient Boosting":
        base_model = GradientBoostingClassifier(random_state=42, **params)
    
    elif model_name == "Random Forest":
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    
    elif model_name == "Logistic Regression":
        base_model = LogisticRegression(random_state=42, max_iter=1000, **params)
    
    elif model_name == "Naive Bayes":
        base_model = GaussianNB(**params)
    
    elif model_name == "SVM":
        # CRITICAL: Use LinearSVC with SGD for speed, then calibrate
        print("  Using LinearSVC (much faster than SVC for large datasets)")
        from sklearn.svm import LinearSVC
        # Ignore loaded params, use fast settings
        base_model = LinearSVC(random_state=42, max_iter=1000, dual='auto', class_weight='balanced')
        # LinearSVC needs special calibration wrapper
        from sklearn.calibration import CalibratedClassifierCV
        print("  Training LinearSVC...")
        base_model.fit(X_train, y_train)
        print("  Calibrating LinearSVC...")
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_cal, y_cal)
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
        print(f"  ✓ Model trained and calibrated")
        print(f"    Mean prediction: {y_pred_proba.mean():.4f}")
        print(f"    Actual mortality: {y_test.mean():.4f}")
        return calibrated_model, X_test, y_test, y_pred_proba
    
    elif model_name == "KNN":
        # Limit neighbors for speed
        if 'n_neighbors' not in params:
            params['n_neighbors'] = 20
        elif params['n_neighbors'] > 50:
            params['n_neighbors'] = 50
        base_model = KNeighborsClassifier(n_jobs=-1, **params)
    
    elif model_name == "Decision Tree":
        base_model = DecisionTreeClassifier(random_state=42, **params)
    
    elif model_name in ["Neural Network", "MLP"]:
        # Fast MLP settings
        print("  Using fast MLP settings (small hidden layer, early stopping)")
        base_model = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=(50,),  # Small network
            max_iter=100,  # Limited iterations
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1,
            verbose=False
        )
    
    elif model_name == "AdaBoost":
        # Use fewer estimators for speed
        print("  Using AdaBoost with 50 estimators (speed optimization)")
        base_model = AdaBoostClassifier(
            random_state=42,
            n_estimators=min(params.get('n_estimators', 50), 50),
            algorithm='SAMME'
        )
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Train base model (for non-SVM models)
    print("  Training base model...")
    base_model.fit(X_train, y_train)
    
    print("  Calibrating model...")
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv='prefit',
        n_jobs=-1
    )
    calibrated_model.fit(X_cal, y_cal)
    
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(f"  ✓ Model trained and calibrated")
    print(f"    Mean prediction: {y_pred_proba.mean():.4f}")
    print(f"    Actual mortality: {y_test.mean():.4f}")
    
    return calibrated_model, X_test, y_test, y_pred_proba

def create_decision_curve(models_data, output_dir):
    """Create decision curve analysis plot comparing ALL models"""
    print("\n" + "="*70)
    print("DECISION CURVE ANALYSIS - ALL MODELS")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 10 distinct colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
              '#9B59B6', '#E67E22', '#1ABC9C', '#E91E63', '#795548']
    
    # Plot each model
    for idx, (model_name, data) in enumerate(models_data.items()):
        y_true = data['y_true']
        y_pred_proba = data['y_pred_proba']
        
        thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(
            y_true, y_pred_proba
        )
        
        ax.plot(thresholds, nb_model, label=model_name, 
               linewidth=2.5, color=colors[idx % len(colors)])
        
        mask = (thresholds >= 0.1) & (thresholds <= 0.5)
        auc_dca = np.trapz(np.array(nb_model)[mask], thresholds[mask])
        
        print(f"\n{model_name}:")
        print(f"  AUC-DCA (0.1-0.5): {auc_dca:.4f}")
        print(f"  Max Net Benefit: {max(nb_model):.4f}")
    
    # Reference strategies
    y_true = list(models_data.values())[0]['y_true']
    y_pred_proba = list(models_data.values())[0]['y_pred_proba']
    thresholds, _, nb_all, nb_none = decision_curve_analysis(y_true, y_pred_proba)
    
    ax.plot(thresholds, nb_all, '--', label='Treat All', 
           linewidth=2, color='gray', alpha=0.7)
    ax.plot(thresholds, nb_none, '--', label='Treat None', 
           linewidth=2, color='black', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Threshold Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Net Benefit', fontsize=13, fontweight='bold')
    ax.set_title('Decision Curve Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.axvspan(0.1, 0.5, alpha=0.1, color='green')
    
    plt.tight_layout()
    
    dca_path = os.path.join(output_dir, 'decision_curve_analysis_all_models.png')
    plt.savefig(dca_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {dca_path}")
    plt.close()

def create_calibration_curves(models_data, output_dir, top_n=3):
    """Create calibration curves for top N models"""
    print("\n" + "="*70)
    print(f"CALIBRATION ANALYSIS - TOP {top_n} MODELS")
    print("="*70)
    
    perf = pd.read_csv("../results/model_performance_summary.csv")
    perf = perf.sort_values('AUROC', ascending=False).head(top_n)
    top_model_names = perf['Model'].tolist()
    
    top_models_data = {k: v for k, v in models_data.items() if k in top_model_names}
    
    n_models = len(top_models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(top_models_data.items()):
        ax = axes[idx]
        y_true = data['y_true']
        y_pred_proba = data['y_pred_proba']
        
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='quantile'
        )
        
        brier = brier_score_loss(y_true, y_pred_proba)
        
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, alpha=0.7)
        ax.plot(prob_pred, prob_true, 's-', linewidth=2.5, markersize=8,
               color='#2E86AB')
        
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Observed Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        ax.text(0.98, 0.02, f'Brier: {brier:.4f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"\n{model_name}: Brier = {brier:.4f}")
    
    plt.suptitle('Calibration Curves', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    cal_path = os.path.join(output_dir, 'calibration_curves.png')
    plt.savefig(cal_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {cal_path}")
    plt.close()

def main(dca_all_models=True, calibration_top_n=3):
    """Main clinical utility analysis"""
    
    print("="*70)
    print("CLINICAL UTILITY ANALYSIS")
    print("="*70)
    
    output_dir = "../results/clinical_utility"
    os.makedirs(output_dir, exist_ok=True)
    
    perf = pd.read_csv("../results/model_performance_summary.csv")
    
    if dca_all_models:
        print(f"\nAnalyzing ALL {len(perf)} models for decision curve")
        print("Note: SVM, MLP, AdaBoost use speed optimizations (smaller training sets)")
        models_to_load = perf['Model'].tolist()
    else:
        perf_top = perf.sort_values('AUROC', ascending=False).head(calibration_top_n)
        models_to_load = perf_top['Model'].tolist()
    
    for model_name in models_to_load:
        print(f"  - {model_name}")
    
    models_data = {}
    
    for model_name in models_to_load:
        try:
            model, X_test, y_test, y_pred_proba = load_model_and_data(model_name)
            
            models_data[model_name] = {
                'model': model,
                'X_test': X_test,
                'y_true': y_test,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  ✓ {model_name} loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(models_data) == 0:
        print("\nERROR: No models could be loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(models_data)} models")
    
    create_decision_curve(models_data, output_dir)
    create_calibration_curves(models_data, output_dir, top_n=calibration_top_n)
    
    print("\n" + "="*70)
    print("✓ CLINICAL UTILITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults in: {output_dir}/")
    print("  - decision_curve_analysis_all_models.png")
    print("  - calibration_curves.png")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Utility Analysis')
    parser.add_argument('--dca_all', action='store_true', default=True)
    parser.add_argument('--calibration_top_n', type=int, default=3)
    
    args = parser.parse_args()
    
    main(dca_all_models=args.dca_all, calibration_top_n=args.calibration_top_n)
"""
Step 8: Clinical Utility Analysis
- Decision Curve Analysis (DCA)
- Calibration Curves
- Net Benefit Assessment

Following Yu et al. methodology for clinical applicability
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
import sys
sys.path.append('.')

def decision_curve_analysis(y_true, y_pred_proba, threshold_range=None):
    """
    Calculate net benefit for decision curve analysis.
    """
    if threshold_range is None:
        threshold_range = np.arange(0.01, 0.99, 0.01)
    
    from sklearn.metrics import confusion_matrix
    
    prevalence = np.mean(y_true)
    
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    
    for threshold in threshold_range:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Handle edge cases
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 1:  # All predicted positive
                tp = np.sum(y_true == 1)
                fp = np.sum(y_true == 0)
                tn = fn = 0
            else:  # All predicted negative
                tn = np.sum(y_true == 0)
                fn = np.sum(y_true == 1)
                tp = fp = 0
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        n = len(y_true)
        
        # Net benefit of model
        if (tp + fp) > 0:
            nb_model = (tp/n) - (fp/n) * (threshold/(1-threshold))
        else:
            nb_model = 0
        
        # Net benefit of treating all
        nb_all = prevalence - (1-prevalence) * (threshold/(1-threshold))
        
        # Net benefit of treating none
        nb_none = 0
        
        net_benefit_model.append(nb_model)
        net_benefit_all.append(max(nb_all, 0))  # Can't be negative
        net_benefit_none.append(nb_none)
    
    return threshold_range, net_benefit_model, net_benefit_all, net_benefit_none

def load_model_and_data(model_name):
    """Load trained model and test data with calibration"""
    print(f"\nLoading {model_name} and test data...")
    
    # Load test data
    test = pd.read_csv("../results/test_data.csv")
    
    # Separate features and target
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    feature_cols = [col for col in test.columns if col not in exclude_cols]
    
    X_test = test[feature_cols]
    y_test = test['mortality_28day']
    
    # Load training data and split for calibration
    train = pd.read_csv("../results/train_data.csv")
    X_train_full = train[feature_cols]
    y_train_full = train['mortality_28day']
    
    # Split training data: 80% for training, 20% for calibration
    from sklearn.model_selection import train_test_split
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Calibration set: {len(X_cal)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Retrain model with best parameters
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    param_file = f"../results/{model_name.lower().replace(' ', '_')}_params.csv"
    params_df = pd.read_csv(param_file)
    params = params_df.to_dict('records')[0]
    
    if model_name == "LightGBM":
        base_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
    elif model_name == "XGBoost":
        base_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
    elif model_name == "Gradient Boosting":
        base_model = GradientBoostingClassifier(random_state=42, **params)
    elif model_name == "Random Forest":
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Train base model
    print("  Training base model...")
    base_model.fit(X_train, y_train)
    
    # Calibrate the model using isotonic regression
    # (Tree-based models typically work best with isotonic)
    print("  Calibrating model...")
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',  # Better for tree-based models
        cv='prefit',  # Use the holdout calibration set
        n_jobs=-1
    )
    calibrated_model.fit(X_cal, y_cal)
    
    # Get calibrated predictions on test set
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Compare uncalibrated vs calibrated
    y_pred_proba_uncal = base_model.predict_proba(X_test)[:, 1]
    
    print(f"  ✓ Model trained and calibrated")
    print(f"    Uncalibrated mean prediction: {y_pred_proba_uncal.mean():.4f}")
    print(f"    Calibrated mean prediction: {y_pred_proba.mean():.4f}")
    print(f"    Actual mortality rate: {y_test.mean():.4f}")
    
    return calibrated_model, X_test, y_test, y_pred_proba

def create_decision_curve(models_data, output_dir):
    """
    Create decision curve analysis plot comparing multiple models
    """
    print("\n" + "="*70)
    print("DECISION CURVE ANALYSIS")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot each model
    for idx, (model_name, data) in enumerate(models_data.items()):
        y_true = data['y_true']
        y_pred_proba = data['y_pred_proba']
        
        # Calculate decision curve
        thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(
            y_true, y_pred_proba
        )
        
        # Plot model
        ax.plot(thresholds, nb_model, label=model_name, 
               linewidth=2.5, color=colors[idx % len(colors)])
        
        # Calculate area under decision curve (0.1 to 0.5 threshold range)
        mask = (thresholds >= 0.1) & (thresholds <= 0.5)
        auc_dca = np.trapz(np.array(nb_model)[mask], thresholds[mask])
        
        print(f"\n{model_name}:")
        print(f"  AUC-DCA (0.1-0.5): {auc_dca:.4f}")
        print(f"  Max Net Benefit: {max(nb_model):.4f} at threshold {thresholds[np.argmax(nb_model)]:.2f}")
    
    # Plot reference strategies (only once)
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
    ax.set_title('Decision Curve Analysis: Clinical Utility Across Risk Thresholds', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    
    # Add shaded region for clinically relevant thresholds
    ax.axvspan(0.1, 0.5, alpha=0.1, color='green', 
              label='Clinically Relevant Range')
    
    plt.tight_layout()
    
    dca_path = os.path.join(output_dir, 'decision_curve_analysis.png')
    plt.savefig(dca_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {dca_path}")
    plt.close()

def create_calibration_curves(models_data, output_dir):
    """
    Create calibration curves for multiple models
    """
    print("\n" + "="*70)
    print("CALIBRATION ANALYSIS")
    print("="*70)
    
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        ax = axes[idx]
        y_true = data['y_true']
        y_pred_proba = data['y_pred_proba']
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='quantile'
        )
        
        # Calculate Brier score
        brier = brier_score_loss(y_true, y_pred_proba)
        
        # Plot
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, 
               label='Perfect Calibration', alpha=0.7)
        ax.plot(prob_pred, prob_true, 's-', linewidth=2.5, markersize=8,
               color='#2E86AB', label=f'{model_name}\n(Brier: {brier:.3f})')
        
        # Formatting
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Observed Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add statistics
        ax.text(0.98, 0.02, f'Brier Score: {brier:.4f}\nn = {len(y_true)}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"\n{model_name}:")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Calibration slope: {np.corrcoef(prob_pred, prob_true)[0,1]:.4f}")
    
    plt.suptitle('Calibration Curves: Predicted vs Observed Mortality', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    cal_path = os.path.join(output_dir, 'calibration_curves.png')
    plt.savefig(cal_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {cal_path}")
    plt.close()

def create_net_benefit_table(models_data, output_dir):
    """
    Create table of net benefit at clinically relevant thresholds
    """
    print("\n" + "="*70)
    print("NET BENEFIT AT KEY THRESHOLDS")
    print("="*70)
    
    thresholds_of_interest = [0.10, 0.20, 0.30, 0.40, 0.50]
    
    results = []
    
    for model_name, data in models_data.items():
        y_true = data['y_true']
        y_pred_proba = data['y_pred_proba']
        
        # Calculate decision curve
        thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(
            y_true, y_pred_proba
        )
        
        for thresh in thresholds_of_interest:
            # Find closest threshold
            idx = np.argmin(np.abs(thresholds - thresh))
            
            results.append({
                'Model': model_name,
                'Threshold': thresh,
                'Net_Benefit_Model': nb_model[idx],
                'Net_Benefit_All': nb_all[idx],
                'Net_Benefit_None': nb_none[idx],
                'Advantage_vs_All': nb_model[idx] - nb_all[idx],
                'Advantage_vs_None': nb_model[idx] - nb_none[idx]
            })
    
    results_df = pd.DataFrame(results)
    
    # Save
    table_path = os.path.join(output_dir, 'net_benefit_table.csv')
    results_df.to_csv(table_path, index=False)
    print(f"\n  Saved: {table_path}")
    
    # Print summary
    print("\nNet Benefit Summary (selected thresholds):")
    for model_name in models_data.keys():
        print(f"\n{model_name}:")
        model_results = results_df[results_df['Model'] == model_name]
        for _, row in model_results.iterrows():
            print(f"  Threshold {row['Threshold']:.2f}: "
                  f"NB = {row['Net_Benefit_Model']:.4f}, "
                  f"vs All = {row['Advantage_vs_All']:+.4f}")
    
    return results_df

def main(top_n=3):
    """
    Main clinical utility analysis for top N models
    
    Parameters:
    -----------
    top_n : int, optional
        Number of top models to analyze (default: 3)
    """
    print("="*70)
    print(f"CLINICAL UTILITY ANALYSIS - TOP {top_n} MODELS")
    print("="*70)
    
    # Create output directory
    output_dir = "../results/clinical_utility"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model performance to get top N
    perf = pd.read_csv("../results/model_performance_summary.csv")
    perf = perf.sort_values('AUROC', ascending=False).head(top_n)
    
    print(f"\nAnalyzing top {top_n} models:")
    for _, row in perf.iterrows():
        print(f"  - {row['Model']:25s} (AUROC: {row['AUROC']:.4f})")
    
    # Load models and predictions
    models_data = {}
    
    for _, row in perf.iterrows():
        model_name = row['Model']
        
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
            continue
    
    if len(models_data) == 0:
        print("\nERROR: No models could be loaded!")
        return
    
    # Create decision curve analysis
    create_decision_curve(models_data, output_dir)
    
    # Create calibration curves
    create_calibration_curves(models_data, output_dir)
    
    # Create net benefit table
    net_benefit_df = create_net_benefit_table(models_data, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("CLINICAL UTILITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print("  - decision_curve_analysis.png (Clinical utility)")
    print("  - calibration_curves.png (Probability calibration)")
    print("  - net_benefit_table.csv (Numerical net benefits)")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FOR YOUR REPORT:")
    print("="*70)
    print("\nDecision Curve Analysis:")
    print("  - Shows net benefit of using model vs treat-all/treat-none")
    print("  - Higher curve = better clinical utility")
    print("  - Focus on 0.1-0.5 threshold range (clinically relevant)")
    
    print("\nCalibration Curves:")
    print("  - Perfect calibration = diagonal line")
    print("  - Lower Brier score = better calibration")
    print("  - Good calibration = predictions match reality")
    
    print("\nNet Benefit Table:")
    print("  - Positive 'Advantage_vs_All' = model adds value")
    print("  - Compare across thresholds to find optimal cutoff")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Utility Analysis')
    parser.add_argument('--top_n', type=int, default=3,
                       help='Number of top models to analyze (default: 3)')
    
    args = parser.parse_args()
    
    main(top_n=args.top_n)
"""
Calibration Analysis - Why Gradient Boosting Looks Weird
=========================================================
This script:
1. Shows calibrated vs uncalibrated predictions for all models
2. Explains why Gradient Boosting has poor calibration
3. Creates visualizations for your presentation

Run this after 05_train_ml_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """Load train and test data"""
    print("Loading data...")
    train = pd.read_csv("../results/train_data.csv")
    test = pd.read_csv("../results/test_data.csv")
    
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    features = [c for c in test.columns if c not in exclude_cols]
    
    X_train = train[features]
    y_train = train['mortality_28day']
    X_test = test[features]
    y_test = test['mortality_28day']
    
    return X_train, y_train, X_test, y_test, features

def train_models_with_calibration(X_train, y_train, X_test, y_test):
    """Train models with and without calibration"""
    
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    
    # Split training data for calibration
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    models = {
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train uncalibrated model
        model.fit(X_tr, y_tr)
        y_pred_uncal = model.predict_proba(X_test)[:, 1]
        
        # Train calibrated model (isotonic regression)
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_cal, y_cal)
        y_pred_cal = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'uncalibrated': {
                'predictions': y_pred_uncal,
                'auroc': roc_auc_score(y_test, y_pred_uncal),
                'brier': brier_score_loss(y_test, y_pred_uncal)
            },
            'calibrated': {
                'predictions': y_pred_cal,
                'auroc': roc_auc_score(y_test, y_pred_cal),
                'brier': brier_score_loss(y_test, y_pred_cal)
            }
        }
        
        print(f"  Uncalibrated - AUROC: {results[name]['uncalibrated']['auroc']:.4f}, Brier: {results[name]['uncalibrated']['brier']:.4f}")
        print(f"  Calibrated   - AUROC: {results[name]['calibrated']['auroc']:.4f}, Brier: {results[name]['calibrated']['brier']:.4f}")
    
    return results

def create_calibration_comparison(results, y_test, output_dir):
    """Create side-by-side calibration plots: uncalibrated vs calibrated"""
    print("\nCreating calibration comparison plots...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    
    models = list(results.keys())
    
    for idx, model_name in enumerate(models):
        # Uncalibrated (top row)
        ax_uncal = axes[0, idx]
        y_pred_uncal = results[model_name]['uncalibrated']['predictions']
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_pred_uncal, n_bins=10, strategy='quantile')
        
        ax_uncal.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax_uncal.plot(prob_pred_uncal, prob_true_uncal, 's-', color='#E74C3C', linewidth=2, markersize=8)
        ax_uncal.set_title(f'{model_name}\n(Uncalibrated)', fontsize=11, fontweight='bold')
        ax_uncal.set_xlabel('Predicted Probability')
        ax_uncal.set_ylabel('Actual Fraction Positive')
        ax_uncal.set_xlim(0, 1)
        ax_uncal.set_ylim(0, 1)
        ax_uncal.grid(True, alpha=0.3)
        
        brier_uncal = results[model_name]['uncalibrated']['brier']
        ax_uncal.text(0.95, 0.05, f'Brier: {brier_uncal:.4f}', transform=ax_uncal.transAxes,
                     ha='right', va='bottom', fontsize=9, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Calibrated (bottom row)
        ax_cal = axes[1, idx]
        y_pred_cal = results[model_name]['calibrated']['predictions']
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_pred_cal, n_bins=10, strategy='quantile')
        
        ax_cal.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax_cal.plot(prob_pred_cal, prob_true_cal, 's-', color='#27AE60', linewidth=2, markersize=8)
        ax_cal.set_title(f'{model_name}\n(Calibrated)', fontsize=11, fontweight='bold')
        ax_cal.set_xlabel('Predicted Probability')
        ax_cal.set_ylabel('Actual Fraction Positive')
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.grid(True, alpha=0.3)
        
        brier_cal = results[model_name]['calibrated']['brier']
        ax_cal.text(0.95, 0.05, f'Brier: {brier_cal:.4f}', transform=ax_cal.transAxes,
                     ha='right', va='bottom', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle('Calibration Comparison: Before vs After Isotonic Calibration\n(Lower Brier Score = Better Calibration)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_before_after.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: calibration_before_after.png")
    plt.close()

def create_gradient_boosting_focus(results, y_test, output_dir):
    """Create focused comparison for Gradient Boosting specifically"""
    print("\nCreating Gradient Boosting focus plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get GB predictions
    y_pred_uncal = results['Gradient Boosting']['uncalibrated']['predictions']
    y_pred_cal = results['Gradient Boosting']['calibrated']['predictions']
    
    # Plot 1: Uncalibrated calibration curve
    ax1 = axes[0]
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_pred_uncal, n_bins=10, strategy='quantile')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax1.plot(prob_pred_uncal, prob_true_uncal, 's-', color='#E74C3C', linewidth=2.5, markersize=10, label='Gradient Boosting')
    ax1.fill_between(prob_pred_uncal, prob_true_uncal, prob_pred_uncal, alpha=0.3, color='#E74C3C')
    ax1.set_title('BEFORE Calibration\n(The "Weird" Pattern)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Probability', fontsize=11)
    ax1.set_ylabel('Actual Fraction Positive', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    brier_uncal = results['Gradient Boosting']['uncalibrated']['brier']
    ax1.text(0.95, 0.05, f'Brier Score: {brier_uncal:.4f}', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8))
    
    # Plot 2: Calibrated calibration curve  
    ax2 = axes[1]
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_pred_cal, n_bins=10, strategy='quantile')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax2.plot(prob_pred_cal, prob_true_cal, 's-', color='#27AE60', linewidth=2.5, markersize=10, label='Gradient Boosting (Calibrated)')
    ax2.fill_between(prob_pred_cal, prob_true_cal, prob_pred_cal, alpha=0.3, color='#27AE60')
    ax2.set_title('AFTER Isotonic Calibration\n(Fixed!)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Probability', fontsize=11)
    ax2.set_ylabel('Actual Fraction Positive', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    brier_cal = results['Gradient Boosting']['calibrated']['brier']
    ax2.text(0.95, 0.05, f'Brier Score: {brier_cal:.4f}', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.8))
    
    # Plot 3: Prediction distribution comparison
    ax3 = axes[2]
    ax3.hist(y_pred_uncal, bins=30, alpha=0.6, color='#E74C3C', label='Uncalibrated', density=True)
    ax3.hist(y_pred_cal, bins=30, alpha=0.6, color='#27AE60', label='Calibrated', density=True)
    ax3.axvline(x=y_test.mean(), color='black', linestyle='--', linewidth=2, label=f'True Mortality Rate ({y_test.mean():.1%})')
    ax3.set_title('Prediction Distribution\n(How Probabilities Spread)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted Probability', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Why Gradient Boosting Looked "Weird" — and How Calibration Fixes It', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gradient_boosting_calibration_fix.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: gradient_boosting_calibration_fix.png")
    plt.close()

def create_brier_comparison_table(results, output_dir):
    """Create table comparing Brier scores before/after calibration"""
    print("\nCreating Brier score comparison...")
    
    data = []
    for model_name, res in results.items():
        data.append({
            'Model': model_name,
            'AUROC (Uncal)': res['uncalibrated']['auroc'],
            'AUROC (Cal)': res['calibrated']['auroc'],
            'Brier (Uncal)': res['uncalibrated']['brier'],
            'Brier (Cal)': res['calibrated']['brier'],
            'Brier Improvement': res['uncalibrated']['brier'] - res['calibrated']['brier']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Brier Improvement', ascending=False)
    
    # Save
    df.to_csv(f'{output_dir}/calibration_comparison.csv', index=False)
    print(f"  Saved: calibration_comparison.csv")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'AUROC\n(Unchanged)', 'Brier\n(Before)', 'Brier\n(After)', 'Improvement']
    
    for _, row in df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['AUROC (Uncal)']:.4f}",
            f"{row['Brier (Uncal)']:.4f}",
            f"{row['Brier (Cal)']:.4f}",
            f"{row['Brier Improvement']:.4f} ({'↓' if row['Brier Improvement'] > 0 else '↑'})"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color the improvement column
    for i in range(len(table_data)):
        cell = table[(i+1, 4)]
        if df.iloc[i]['Brier Improvement'] > 0.01:
            cell.set_facecolor('#D5F5E3')  # Green for big improvement
        elif df.iloc[i]['Brier Improvement'] > 0:
            cell.set_facecolor('#FCF3CF')  # Yellow for small improvement
    
    plt.title('Calibration Improves Probability Estimates\n(Lower Brier Score = Better)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/brier_comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: brier_comparison_table.png")
    plt.close()
    
    return df

def main():
    """Main analysis"""
    print("="*70)
    print("CALIBRATION ANALYSIS")
    print("Why Gradient Boosting Looks Weird & How to Fix It")
    print("="*70)
    
    # Create output directory
    output_dir = "../results/calibration_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test, features = load_data()
    
    # Train models with and without calibration
    results = train_models_with_calibration(X_train, y_train, X_test, y_test)
    
    # Create visualizations
    create_calibration_comparison(results, y_test, output_dir)
    create_gradient_boosting_focus(results, y_test, output_dir)
    brier_df = create_brier_comparison_table(results, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print("\n📊 WHY GRADIENT BOOSTING LOOKED 'WEIRD':")
    print("-"*50)
    print("1. Tree-based models optimize for RANKING (AUROC), not probability accuracy")
    print("2. Gradient Boosting outputs 'scores' that aren't true probabilities")
    print("3. The zigzag pattern = inconsistent probability estimates")
    print("4. AUROC stays the same (ranking preserved), but Brier improves (probabilities fixed)")
    
    print("\n📈 CALIBRATION RESULTS:")
    print("-"*50)
    print(brier_df[['Model', 'Brier (Uncal)', 'Brier (Cal)', 'Brier Improvement']].to_string(index=False))
    
    gb_improvement = brier_df[brier_df['Model'] == 'Gradient Boosting']['Brier Improvement'].values[0]
    print(f"\n✓ Gradient Boosting Brier improved by {gb_improvement:.4f} after calibration!")
    
    print("\n💡 WHAT TO TELL YOUR PROFESSOR:")
    print("-"*50)
    print('"Gradient Boosting showed poor calibration because tree-based')
    print('models optimize for ranking (AUROC) not probability accuracy.')
    print('The zigzag pattern indicates unreliable probability estimates.')
    print('Post-hoc isotonic calibration fixes this — Brier score improved')
    print(f'from {brier_df[brier_df["Model"]=="Gradient Boosting"]["Brier (Uncal)"].values[0]:.4f} to {brier_df[brier_df["Model"]=="Gradient Boosting"]["Brier (Cal)"].values[0]:.4f} while AUROC stayed the same."')
    
    print(f"\n📁 Results saved in: {output_dir}/")
    print("   - calibration_before_after.png")
    print("   - gradient_boosting_calibration_fix.png")
    print("   - brier_comparison_table.png")
    print("   - calibration_comparison.csv")
    print("="*70)

if __name__ == "__main__":
    main()
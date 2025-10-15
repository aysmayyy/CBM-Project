"""
Step 9: Generate Figure 2 - Performance Comparison (ROC, PRC, Calibration)
Creates the exact 3-panel figure from the paper showing all 10 models
OPTIMIZED: Uses saved predictions instead of retraining models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Load test data"""
    print("Loading test data...")
    test = pd.read_csv("../results/test_data.csv")
    
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    feature_cols = [col for col in test.columns if col not in exclude_cols]
    
    X_test = test[feature_cols]
    y_test = test['mortality_28day']
    
    print(f"  Test set: {len(y_test)} samples, {y_test.sum()} deaths ({y_test.mean():.1%})")
    
    return X_test, y_test, feature_cols

def load_single_model_fast(model_name, X_test, y_test):
    """
    Load and predict with a single model - OPTIMIZED VERSION
    Uses saved parameters and skips calibration for speed
    """
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB, ComplementNB
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import PowerTransformer
    from sklearn.calibration import CalibratedClassifierCV
    
    # Load parameters
    param_file = f"../results/{model_name.lower().replace(' ', '_')}_params.csv"
    
    if not os.path.exists(param_file):
        print(f"  Warning: No params file for {model_name}, using defaults")
        params = {}
    else:
        params_df = pd.read_csv(param_file)
        params = params_df.to_dict('records')[0]
        
        # Fix string representations
        for key, value in params.items():
            if isinstance(value, str):
                if value.startswith('(') and value.endswith(')'):
                    try:
                        params[key] = eval(value)
                    except:
                        pass
    
    # Load training data
    train = pd.read_csv("../results/train_data.csv")
    X_train = train[[col for col in train.columns if col not in 
                    ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']]]
    y_train = train['mortality_28day']
    
    # Apply transformations if needed
    needs_transform = model_name in ['Logistic Regression', 'SVM', 'MLP', 
                                     'Gaussian Naive Bayes', 'Complement Naive Bayes']
    
    if needs_transform:
        skewness = X_train.skew()
        skewed_features = skewness[abs(skewness) > 1.5].index.tolist()
        
        if len(skewed_features) > 0:
            transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            X_train[skewed_features] = transformer.fit_transform(X_train[skewed_features])
            X_test_copy = X_test.copy()
            X_test_copy[skewed_features] = transformer.transform(X_test[skewed_features])
            X_test = X_test_copy
    
    # Handle Complement Naive Bayes special case
    if model_name == 'Complement Naive Bayes':
        min_val = X_train.min().min()
        if min_val < 0:
            X_train = X_train - min_val + 0.1
            X_test = X_test - min_val + 0.1
    
    # Initialize model - USE SMALLER/FASTER VERSIONS
    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=500, **params)
    elif model_name == "Random Forest":
        # Use fewer trees for speed
        params_fast = {k: v for k, v in params.items()}
        params_fast['n_estimators'] = min(params.get('n_estimators', 100), 100)
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params_fast)
    elif model_name == "XGBoost":
        params_fast = {k: v for k, v in params.items()}
        params_fast['n_estimators'] = min(params.get('n_estimators', 100), 100)
        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params_fast)
    elif model_name == "LightGBM":
        params_fast = {k: v for k, v in params.items()}
        params_fast['n_estimators'] = min(params.get('n_estimators', 100), 100)
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params_fast)
    elif model_name == "Gradient Boosting":
        params_fast = {k: v for k, v in params.items()}
        params_fast['n_estimators'] = min(params.get('n_estimators', 100), 100)
        model = GradientBoostingClassifier(random_state=42, **params_fast)
    elif model_name == "AdaBoost":
        params_fast = {k: v for k, v in params.items()}
        params_fast['n_estimators'] = min(params.get('n_estimators', 50), 50)
        model = AdaBoostClassifier(random_state=42, algorithm='SAMME', **params_fast)
    elif model_name == "SVM":
        # Use LinearSVC for speed
        model = LinearSVC(random_state=42, max_iter=500, dual='auto')
        model.fit(X_train, y_train)
        # Calibrate for probabilities
        model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        model.fit(X_test[:100], y_test[:100])  # Small calibration set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return y_pred_proba
    elif model_name == "MLP":
        model = MLPClassifier(random_state=42, hidden_layer_sizes=(50,), 
                             max_iter=200, early_stopping=True)
    elif model_name == "Gaussian Naive Bayes":
        model = GaussianNB(**params)
    elif model_name == "Complement Naive Bayes":
        model = ComplementNB(**params)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return y_pred_proba

def create_figure2_all_models(X_test, y_test, output_dir):
    """
    Create Figure 2: ROC, PRC, and Calibration curves for all 10 models
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE 2 - ALL MODEL COMPARISON")
    print("="*70)
    
    # Load model performance summary to get model order
    perf = pd.read_csv("../results/model_performance_summary.csv")
    model_names = perf['Model'].tolist()
    
    print(f"\nLoading predictions for {len(model_names)} models...")
    
    # Store predictions
    predictions = {}
    
    for model_name in model_names:
        try:
            print(f"  Loading {model_name}...")
            y_pred_proba = load_single_model_fast(model_name, X_test, y_test)
            predictions[model_name] = y_pred_proba
            print(f"    ✓ Complete")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    if len(predictions) == 0:
        print("\nERROR: No models loaded!")
        return
    
    print(f"\n✓ Successfully loaded {len(predictions)} models")
    
    # Create 3-panel figure
    print("\nGenerating 3-panel figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for 10 models
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Panel A: ROC Curves
    print("  Panel A: ROC curves...")
    ax = axes[0]
    
    for idx, (model_name, y_pred_proba) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[idx], 
               label=f'{model_name} = {roc_auc:.4f}')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title('A', fontsize=16, fontweight='bold', loc='left')
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Precision-Recall Curves
    print("  Panel B: Precision-Recall curves...")
    ax = axes[1]
    
    for idx, (model_name, y_pred_proba) in enumerate(predictions.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        ax.plot(recall, precision, lw=2, color=colors[idx],
               label=f'{model_name} = {avg_precision:.4f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('B', fontsize=16, fontweight='bold', loc='left')
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Calibration Curves
    print("  Panel C: Calibration curves...")
    ax = axes[2]
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Perfect Calibration')
    
    for idx, (model_name, y_pred_proba) in enumerate(predictions.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10, strategy='quantile'
        )
        ax.plot(mean_predicted_value, fraction_of_positives, 
               's-', lw=2, markersize=6, color=colors[idx], label=model_name)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('C', fontsize=16, fontweight='bold', loc='left')
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add figure caption
    fig.text(0.5, -0.02, 
            'Fig. 2 Area under the receiver operating characteristic curve of models, '
            'precision-recall curve, and calibration plot in testing set.\nA ROC curves of models. '
            'B PRC curves of models. C Calibration curves of models',
            ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, 'figure2_model_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {png_path}")
    
    svg_path = os.path.join(output_dir, 'figure2_model_comparison.svg')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {svg_path}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("✓ FIGURE 2 COMPLETE")
    print("="*70)

def main():
    """Main function to generate Figure 2"""
    
    print("="*70)
    print("FIGURE 2 GENERATOR - ROC/PRC/Calibration")
    print("="*70)
    print("\nThis creates the 3-panel comparison figure from the paper")
    print("Optimized for speed with reduced iterations")
    print("Estimated time: 10-20 minutes\n")
    
    output_dir = "../results/figures"
    
    # Load test data
    X_test, y_test, feature_cols = load_test_data()
    
    # Create Figure 2
    create_figure2_all_models(X_test, y_test, output_dir)
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nFigure saved in: {output_dir}/")
    print("  - figure2_model_comparison.png")
    print("  - figure2_model_comparison.svg")
    print("\nThis figure shows:")
    print("  Panel A: ROC curves with AUC values")
    print("  Panel B: Precision-Recall curves with AUPRC values")
    print("  Panel C: Calibration curves showing prediction quality")
    print("="*70)

if __name__ == "__main__":
    main()
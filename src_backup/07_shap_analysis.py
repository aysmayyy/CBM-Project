#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Step 7: SHAP Analysis for Model Interpretability
Explains predictions from the best performing model (LightGBM)
Following Yu et al. methodology
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shap
import joblib
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Import your model configs
import sys
sys.path.append('.')


# In[2]:


def load_top_models_and_data(top_n=3):
    """Load the top N models and test data"""
    print("Loading model performance results...")

    # Load performance summary to identify top models
    perf = pd.read_csv("../results/models/model_performance_summary.csv")
    perf = perf.sort_values('AUROC', ascending=False)

    top_models = perf.head(top_n)

    print(f"\nTop {top_n} models by AUROC:")
    for idx, row in top_models.iterrows():
        print(f"  {row['Model']:25s} AUROC: {row['AUROC']:.4f}")

    # Load test data
    test = pd.read_csv("../results/data/test_data.csv")

    # Separate features and target
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    feature_cols = [col for col in test.columns if col not in exclude_cols]

    X_test = test[feature_cols]
    y_test = test['mortality_28day']

    print(f"\nTest set: {len(X_test)} samples, {len(feature_cols)} features")
    print(f"Mortality rate: {y_test.mean():.1%}")

    return top_models, X_test, y_test, feature_cols


# In[3]:


def retrain_model(model_name, X_train, y_train):
    """Retrain a model for SHAP analysis"""
    print(f"\n{'='*70}")
    print(f"Retraining {model_name}")
    print(f"{'='*70}")

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB, ComplementNB

    # Load best params
    param_file = f"../results/models/{model_name.lower().replace(' ', '_')}_params.csv"

    try:
        params_df = pd.read_csv(param_file)
        params = params_df.to_dict('records')[0]
        print(f"  Loaded parameters from {param_file}")
    except:
        print(f"  Warning: Could not load params file, using defaults")
        params = {}

    # Create model based on name
    if model_name == "LightGBM":
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42, **params)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=42, **params)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000, **params)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42, **params)
    elif model_name == "MLP":
        model = MLPClassifier(random_state=42, max_iter=500, **params)
    elif model_name == "Gaussian Naive Bayes":
        model = GaussianNB(**params)
    elif model_name == "Complement Naive Bayes":
        model = ComplementNB(**params)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Train the model
    print("  Training model...")
    model.fit(X_train, y_train)
    print("  Training complete!")

    return model


# In[4]:


def generate_shap_analysis(model, X_test, y_test, model_name, output_dir, sample_size=500):
    """
    Generate comprehensive SHAP analysis

    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels
    model_name : str
        Name of the model
    output_dir : str
        Directory to save plots
    sample_size : int
        Number of samples for SHAP (use less for speed)
    """
    print(f"\n{'='*70}")
    print(f"Generating SHAP Analysis: {model_name}")
    print(f"{'='*70}")

    # Sample data for faster computation
    if len(X_test) > sample_size:
        print(f"  Using {sample_size} samples for SHAP analysis...")
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
    else:
        X_sample = X_test
        y_sample = y_test

    # Determine if tree-based or linear model
    tree_based = model_name in ["LightGBM", "XGBoost", "Gradient Boosting", 
                                 "Random Forest", "AdaBoost"]

    try:
        # Create appropriate explainer
        if tree_based:
            print("  Creating TreeExplainer (fast)...")
            explainer = shap.TreeExplainer(model)
        else:
            print("  Creating KernelExplainer (slower for linear models)...")
            # For linear models, use KernelExplainer with smaller background
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict_proba, background)

        # Calculate SHAP values
        print("  Computing SHAP values...")
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, get values for positive class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]  # Positive class (mortality = 1)
        elif isinstance(shap_values, list):
            shap_values = shap_values[0]

        print(f"  SHAP values computed! Shape: {shap_values.shape}")

    except Exception as e:
        print(f"  ERROR: Could not generate SHAP for {model_name}: {e}")
        print(f"  Skipping this model...")
        return None, None

    # Create output directory for this model
    model_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)

    # 1. Feature Importance Bar Plot
    print("  Creating bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f'{model_name}: Feature Importance (SHAP)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mean |SHAP value| (Average impact on model output)', fontsize=11)
    plt.tight_layout()
    bar_path = os.path.join(model_dir, 'shap_bar.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SHAP Summary Plot (beeswarm)
    print("  Creating summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f'{model_name}: Feature Impact on Predictions', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP value (impact on model output)', fontsize=11)
    plt.tight_layout()
    summary_path = os.path.join(model_dir, 'shap_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Combine both plots side by side
    print("  Creating combined plot...")
    img1 = Image.open(bar_path)
    img2 = Image.open(summary_path)

    # Create combined image
    width = img1.width + img2.width + 100
    height = max(img1.height, img2.height) + 100
    combined = Image.new('RGB', (width, height), color='white')
    combined.paste(img1, (50, 50))
    combined.paste(img2, (img1.width + 50, 50))

    combined_path = os.path.join(model_dir, 'shap_combined.png')
    combined.save(combined_path)

    # 4. Top feature dependence plots (only for tree models - faster)
    if tree_based:
        print("  Creating dependence plots for top 4 features...")

        shap_importance = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(shap_importance)[-4:][::-1]
        top_features = [X_sample.columns[i] for i in top_features_idx]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for idx, feat in enumerate(top_features):
            shap.dependence_plot(feat, shap_values, X_sample, ax=axes[idx], show=False)
            axes[idx].set_title(f'{feat}', fontsize=12, fontweight='bold')

        plt.suptitle(f'{model_name}: Top 4 Feature Dependence Plots', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        dep_path = os.path.join(model_dir, 'dependence_plots.png')
        plt.savefig(dep_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Save SHAP values and feature importance
    print("  Saving numerical results...")
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_df['actual_outcome'] = y_sample.values

    importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    csv_path = os.path.join(model_dir, 'shap_values.csv')
    shap_df.to_csv(csv_path, index=False)

    importance_path = os.path.join(model_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)

    # Print top 10 features
    print("\n  TOP 10 MOST IMPORTANT FEATURES:")
    print("  " + "-"*66)
    for idx, row in importance_df.head(10).iterrows():
        direction = "↑" if row['mean_shap'] > 0 else "↓"
        print(f"  {direction} {row['feature']:28s} | Impact: {row['mean_abs_shap']:.4f}")

    print(f"\n  Results saved in: {model_dir}/")

    return shap_values, importance_df


# In[5]:


def main(top_n=3):
    """
    Main SHAP analysis pipeline for top N models

    Parameters:
    -----------
    top_n : int, optional
        Number of top models to analyze (default: 3)
    """
    print("="*70)
    print(f"SHAP ANALYSIS FOR TOP {top_n} MODELS")
    print("="*70)

    # Create output directory
    output_dir = "../results/shap_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Load top N models and data
    top_models, X_test, y_test, feature_cols = load_top_models_and_data(top_n=top_n)

    # Load training data
    print("\nLoading training data...")
    train = pd.read_csv("../results/data/train_data.csv")
    X_train = train[feature_cols]
    y_train = train['mortality_28day']
    print(f"Training set: {len(X_train)} samples")

    # Analyze each of the top 3 models
    results_summary = []

    for idx, row in top_models.iterrows():
        model_name = row['Model']
        auroc = row['AUROC']

        try:
            # Retrain the model
            model = retrain_model(model_name, X_train, y_train)

            # Generate SHAP analysis
            shap_values, importance_df = generate_shap_analysis(
                model, X_test, y_test, model_name, output_dir, sample_size=500
            )

            if importance_df is not None:
                results_summary.append({
                    'model': model_name,
                    'auroc': auroc,
                    'top_features': importance_df.head(5)['feature'].tolist()
                })

        except Exception as e:
            print(f"\n✗ ERROR analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "="*70)
    print("SHAP ANALYSIS COMPLETE!")
    print("="*70)

    print(f"\nAnalyzed {len(results_summary)} models successfully")
    print(f"\nResults saved in: {output_dir}/")

    for result in results_summary:
        model_name = result['model']
        print(f"\n  {model_name} (AUROC: {result['auroc']:.4f})")
        print(f"    Top 5 features: {', '.join(result['top_features'][:5])}")
        print(f"    Files: {output_dir}/{model_name.replace(' ', '_')}/")

    print("\n" + "="*70)
    print("Next steps:")
    print("  - Review SHAP plots to understand feature importance")
    print("  - Compare top features across the 3 models")
    print("  - Use these insights for your Discussion section")
    print("="*70)


# In[6]:


main(top_n=3)


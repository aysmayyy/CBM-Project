#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Extended Threshold Analysis - CBM Project Extension
===================================================
Expands on basic threshold analysis by:
1. Running analysis on TOP 5 models (not just LightGBM)
2. Adding more clinical metrics (PPV, NPV, F1, Balanced Accuracy, Youden Index)
3. Creating publication-quality comparison visualizations
4. Finding optimal threshold per model using multiple criteria

For 10-15 minute presentation on Extension
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score,
                             precision_score, recall_score, balanced_accuracy_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (12, 8)


# In[2]:


def load_data():
    """Load train and test data"""
    print("Loading data...")
    train = pd.read_csv("../results/data/train_data.csv")
    test = pd.read_csv("../results/data/test_data.csv")

    # Get feature columns
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
    features = [c for c in test.columns if c not in exclude_cols]

    X_train = train[features]
    y_train = train['mortality_28day']
    X_test = test[features]
    y_test = test['mortality_28day']

    print(f"Training: {len(X_train)} patients, {y_train.sum()} deaths ({y_train.mean():.1%})")
    print(f"Test: {len(X_test)} patients, {y_test.sum()} deaths ({y_test.mean():.1%})")

    return X_train, y_train, X_test, y_test, features


# In[3]:


def get_top_models(n=5):
    """Get the top N models by AUROC from previous analysis"""
    perf = pd.read_csv("../results/models/model_performance_summary.csv")
    perf = perf.sort_values('AUROC', ascending=False)
    top_models = perf.head(n)['Model'].tolist()

    print(f"\nTop {n} models by AUROC:")
    for i, row in perf.head(n).iterrows():
        print(f"  {row['Model']:25s} AUROC: {row['AUROC']:.4f}")

    return top_models


# In[4]:


def train_and_predict(model_name, X_train, y_train, X_test):
    """
    Train model and return predictions on test set.
    Matches preprocessing from 05_train_ml_models.ipynb
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB, ComplementNB
    from sklearn.preprocessing import PowerTransformer

    # Make copies
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Models that need Yeo-Johnson transformation
    needs_transform = ['Logistic Regression', 'SVM', 'MLP', 'Gaussian Naive Bayes', 'Complement Naive Bayes']

    # Apply Yeo-Johnson transformation for linear models
    if model_name in needs_transform:
        # Find skewed features (|skewness| > 1.5)
        skewness = X_train.skew()
        skewed_features = skewness[abs(skewness) > 1.5].index.tolist()

        if skewed_features:
            transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            X_train[skewed_features] = transformer.fit_transform(X_train[skewed_features])
            X_test[skewed_features] = transformer.transform(X_test[skewed_features])

    # Load best params if available
    param_file = f"../results/models/{model_name.lower().replace(' ', '_')}_params.csv"
    try:
        params = pd.read_csv(param_file).to_dict('records')[0]
        # Convert float params to int where needed
        int_params = ['n_estimators', 'max_depth', 'num_leaves', 'min_samples_split', 'min_samples_leaf']
        for p in int_params:
            if p in params and pd.notna(params[p]):
                params[p] = int(params[p])
    except:
        params = {}

    # Special handling for Complement NB - shift BOTH train and test
    if model_name == "Complement Naive Bayes":
        min_val = X_train.min().min()
        if min_val < 0:
            shift = -min_val + 0.1
            X_train = X_train + shift
            X_test = X_test + shift
        model = ComplementNB(**{k:v for k,v in params.items() if k in ['alpha', 'norm']})
    elif model_name == "LightGBM":
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42, **params)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif model_name == "Logistic Regression":
        params.pop('solver', None)
        model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', **params)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42, **params)
    elif model_name == "MLP":
        if 'hidden_layer_sizes' in params and isinstance(params['hidden_layer_sizes'], str):
            params['hidden_layer_sizes'] = eval(params['hidden_layer_sizes'])
        model = MLPClassifier(random_state=42, max_iter=500, **params)
    elif model_name == "Gaussian Naive Bayes":
        model = GaussianNB()
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=42, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train and predict
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return y_pred_proba


# In[5]:


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Calculate comprehensive metrics at a given threshold"""
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate all metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    youden_index = sensitivity + specificity - 1

    return {
        'Threshold': threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1_Score': f1,
        'Balanced_Accuracy': balanced_acc,
        'Youden_Index': youden_index,
        'True_Positive': tp,
        'False_Positive': fp,
        'True_Negative': tn,
        'False_Negative': fn
    }


# In[6]:


def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal threshold using different criteria"""
    thresholds = np.arange(0.05, 0.95, 0.01)

    best_youden = {'threshold': 0.5, 'value': -1}
    best_f1 = {'threshold': 0.5, 'value': -1}
    best_balanced = {'threshold': 0.5, 'value': -1}

    for thresh in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_pred_proba, thresh)

        if metrics['Youden_Index'] > best_youden['value']:
            best_youden = {'threshold': thresh, 'value': metrics['Youden_Index']}

        if metrics['F1_Score'] > best_f1['value']:
            best_f1 = {'threshold': thresh, 'value': metrics['F1_Score']}

        if metrics['Balanced_Accuracy'] > best_balanced['value']:
            best_balanced = {'threshold': thresh, 'value': metrics['Balanced_Accuracy']}

    return {
        'Youden_Index': best_youden,
        'F1_Score': best_f1,
        'Balanced_Accuracy': best_balanced
    }


# In[7]:


def run_threshold_analysis_single_model(model_name, y_pred_proba, y_test, thresholds):
    """Run threshold analysis for a single model"""
    print(f"  Analyzing {model_name}...")

    auroc = roc_auc_score(y_test, y_pred_proba)
    print(f"    AUROC: {auroc:.4f}")

    results = []
    for thresh in thresholds:
        metrics = calculate_metrics_at_threshold(y_test, y_pred_proba, thresh)
        metrics['Model'] = model_name
        metrics['AUROC'] = auroc
        results.append(metrics)

    optimal = find_optimal_thresholds(y_test, y_pred_proba)

    return pd.DataFrame(results), optimal


# In[8]:


def create_comparison_plot(all_results, output_dir):
    """Create side-by-side sensitivity/specificity plots for all models"""
    print("\nCreating comparison plots...")

    models = all_results['Model'].unique()
    n_models = len(models)

    # Dynamic grid for any number of models
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.ravel()

    colors = {'Sensitivity': '#E74C3C', 'Specificity': '#3498DB'}

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = all_results[all_results['Model'] == model]

        ax.plot(model_data['Threshold'], model_data['Sensitivity'], 
                'o-', color=colors['Sensitivity'], linewidth=2, markersize=4, label='Sensitivity')
        ax.plot(model_data['Threshold'], model_data['Specificity'], 
                's-', color=colors['Specificity'], linewidth=2, markersize=4, label='Specificity')

        # Find and mark intersection (Youden optimal)
        best_idx = model_data['Youden_Index'].idxmax()
        best_thresh = model_data.loc[best_idx, 'Threshold']
        ax.axvline(x=best_thresh, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal: {best_thresh:.2f}')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Performance')
        ax.set_title(f'{model}\n(AUROC: {model_data["AUROC"].iloc[0]:.3f})')
        ax.legend(loc='center right', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sensitivity vs Specificity Across Decision Thresholds\nBy Model', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_comparison_all_models.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: threshold_comparison_all_models.png")
    plt.close()


# In[9]:


def create_optimal_threshold_comparison(optimal_thresholds, output_dir):
    """Create bar chart comparing optimal thresholds across models"""
    print("\nCreating optimal threshold comparison...")

    # Prepare data
    data = []
    for model, criteria in optimal_thresholds.items():
        data.append({
            'Model': model,
            'Youden Index': criteria['Youden_Index']['threshold'],
            'F1 Score': criteria['F1_Score']['threshold'],
            'Balanced Accuracy': criteria['Balanced_Accuracy']['threshold']
        })

    df = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.25

    bars1 = ax.bar(x - width, df['Youden Index'], width, label='Youden Index', color='#2ECC71')
    bars2 = ax.bar(x, df['F1 Score'], width, label='F1 Score', color='#E74C3C')
    bars3 = ax.bar(x + width, df['Balanced Accuracy'], width, label='Balanced Accuracy', color='#3498DB')

    ax.set_xlabel('Model')
    ax.set_ylabel('Optimal Threshold')
    ax.set_title('Optimal Decision Threshold by Model and Optimization Criterion', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 0.6)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimal_threshold_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: optimal_threshold_comparison.png")
    plt.close()

    return df


# In[10]:


def create_metrics_heatmap(all_results, optimal_thresholds, output_dir):
    """Create heatmap of all metrics at optimal threshold for each model"""
    print("\nCreating metrics heatmap...")

    summary_data = []

    for model in all_results['Model'].unique():
        model_data = all_results[all_results['Model'] == model]

        # Use the same optimal threshold as summary table
        youden_thresh = optimal_thresholds[model]['Youden_Index']['threshold']

        # Find closest threshold in our data
        threshold_diff = (model_data['Threshold'] - youden_thresh).abs()
        closest_idx = threshold_diff.idxmin()
        best_row = model_data.loc[closest_idx]

        summary_data.append({
            'Model': model,
            'Optimal\nThreshold': youden_thresh,
            'Sensitivity': best_row['Sensitivity'],
            'Specificity': best_row['Specificity'],
            'PPV': best_row['PPV'],
            'NPV': best_row['NPV'],
            'F1 Score': best_row['F1_Score'],
            'Balanced\nAccuracy': best_row['Balanced_Accuracy'],
            'Youden\nIndex': best_row['Youden_Index']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Model')

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(summary_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'Score'})

    ax.set_title('Performance Metrics at Optimal Threshold (Youden Index)\nAcross Top 5 Models', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: metrics_heatmap.png")
    plt.close()

    return summary_df


# In[11]:


def create_clinical_scenario_plot(all_results, output_dir):
    """Show how different thresholds affect clinical outcomes"""
    print("\nCreating clinical scenario analysis...")

    # Use best model (LightGBM likely)
    best_model = all_results.groupby('Model')['AUROC'].mean().idxmax()
    model_data = all_results[all_results['Model'] == best_model].copy()

    # Calculate patients affected at each threshold (assuming 1000 patients)
    n_patients = 1000
    mortality_rate = 0.17  # approximately from your data

    model_data['Deaths'] = int(n_patients * mortality_rate)
    model_data['Survivors'] = int(n_patients * (1 - mortality_rate))
    model_data['Correctly_Identified_Deaths'] = (model_data['Sensitivity'] * model_data['Deaths']).astype(int)
    model_data['Missed_Deaths'] = model_data['Deaths'] - model_data['Correctly_Identified_Deaths']
    model_data['False_Alarms'] = ((1 - model_data['Specificity']) * model_data['Survivors']).astype(int)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Deaths identified vs missed
    ax1 = axes[0]
    ax1.fill_between(model_data['Threshold'], model_data['Correctly_Identified_Deaths'], 
                     color='#2ECC71', alpha=0.7, label='Deaths Correctly Flagged')
    ax1.fill_between(model_data['Threshold'], model_data['Correctly_Identified_Deaths'], 
                     model_data['Deaths'], color='#E74C3C', alpha=0.7, label='Deaths Missed')
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Number of Patients (per 1000)')
    ax1.set_title(f'Clinical Impact: Death Detection\n({best_model})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Trade-off visualization
    ax2 = axes[1]
    ax2.plot(model_data['Threshold'], model_data['Missed_Deaths'], 
             'o-', color='#E74C3C', linewidth=2, label='Missed Deaths (FN)')
    ax2.plot(model_data['Threshold'], model_data['False_Alarms'], 
             's-', color='#F39C12', linewidth=2, label='False Alarms (FP)')

    # Mark optimal threshold
    best_idx = model_data['Youden_Index'].idxmax()
    best_thresh = model_data.loc[best_idx, 'Threshold']
    ax2.axvline(x=best_thresh, color='green', linestyle='--', linewidth=2,
                label=f'Optimal Threshold: {best_thresh:.2f}')

    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Number of Patients (per 1000)')
    ax2.set_title('Clinical Trade-off:\nMissed Deaths vs False Alarms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Clinical Decision Impact Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/clinical_impact_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: clinical_impact_analysis.png")
    plt.close()


# In[12]:


def create_f1_ppv_npv_plot(all_results, output_dir):
    """Create plot showing F1, PPV, NPV across thresholds"""
    print("\nCreating F1/PPV/NPV analysis plot...")

    models = all_results['Model'].unique()
    n_models = len(models)

    # Dynamic grid for any number of models
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.ravel()

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = all_results[all_results['Model'] == model]

        ax.plot(model_data['Threshold'], model_data['F1_Score'], 
                'o-', linewidth=2, markersize=3, label='F1 Score', color='#9B59B6')
        ax.plot(model_data['Threshold'], model_data['PPV'], 
                's-', linewidth=2, markersize=3, label='PPV (Precision)', color='#E67E22')
        ax.plot(model_data['Threshold'], model_data['NPV'], 
                '^-', linewidth=2, markersize=3, label='NPV', color='#1ABC9C')

        # Mark F1-optimal threshold
        best_f1_idx = model_data['F1_Score'].idxmax()
        best_f1_thresh = model_data.loc[best_f1_idx, 'Threshold']
        ax.axvline(x=best_f1_thresh, color='#9B59B6', linestyle='--', alpha=0.5)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model}')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('F1 Score, PPV, and NPV Across Decision Thresholds', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/f1_ppv_npv_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: f1_ppv_npv_analysis.png")
    plt.close()


# In[13]:


def create_summary_table(all_results, optimal_thresholds, output_dir):
    """Create comprehensive summary table"""
    print("\nCreating summary tables...")

    summary_rows = []

    for model in all_results['Model'].unique():
        model_data = all_results[all_results['Model'] == model]
        opt = optimal_thresholds[model]

        # Get metrics at Youden-optimal threshold - find CLOSEST threshold in our data
        youden_thresh = opt['Youden_Index']['threshold']

        # Find the row with the closest threshold value
        threshold_diff = (model_data['Threshold'] - youden_thresh).abs()
        closest_idx = threshold_diff.idxmin()
        youden_row = model_data.loc[closest_idx]

        summary_rows.append({
            'Model': model,
            'AUROC': model_data['AUROC'].iloc[0],
            'Optimal_Threshold_Youden': youden_thresh,
            'Optimal_Threshold_F1': opt['F1_Score']['threshold'],
            'Sensitivity_at_Optimal': youden_row['Sensitivity'],
            'Specificity_at_Optimal': youden_row['Specificity'],
            'PPV_at_Optimal': youden_row['PPV'],
            'NPV_at_Optimal': youden_row['NPV'],
            'F1_at_Optimal': youden_row['F1_Score'],
            'Balanced_Acc_at_Optimal': youden_row['Balanced_Accuracy'],
            'Youden_Index': youden_row['Youden_Index']
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('AUROC', ascending=False)

    # Save
    summary_df.to_csv(f'{output_dir}/extended_threshold_summary.csv', index=False)
    print(f"  Saved: extended_threshold_summary.csv")

    return summary_df


# In[14]:


def main():
    """Main analysis pipeline"""
    print("="*70)
    print("EXTENDED THRESHOLD ANALYSIS")
    print("CBM Project Extension")
    print("="*70)

    # Create output directory
    output_dir = "../results/extended_threshold_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test, features = load_data()

    # Get all 10 models
    top_models = get_top_models(n=10)

    # Define thresholds to test
    thresholds = np.arange(0.05, 0.95, 0.05)

    # Run analysis for each model
    all_results = []
    optimal_thresholds = {}

    print("\n" + "="*70)
    print("Running threshold analysis for each model...")
    print("="*70)

    for model_name in top_models:
        try:
            # Train model
            y_pred_proba = train_and_predict(model_name, X_train, y_train, X_test)
            results, optimal = run_threshold_analysis_single_model(
                model_name, y_pred_proba, y_test, thresholds
            )

            all_results.append(results)
            optimal_thresholds[model_name] = optimal

            print(f"    Optimal thresholds:")
            print(f"      Youden: {optimal['Youden_Index']['threshold']:.2f} (value: {optimal['Youden_Index']['value']:.3f})")
            print(f"      F1:     {optimal['F1_Score']['threshold']:.2f} (value: {optimal['F1_Score']['value']:.3f})")

        except Exception as e:
            print(f"  ERROR with {model_name}: {e}")
            continue

    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Save raw results
    all_results_df.to_csv(f'{output_dir}/all_threshold_results.csv', index=False)
    print(f"\nSaved raw results: all_threshold_results.csv")

    # Create visualizations
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)

    # 1. Comparison plot (sens/spec for all models)
    create_comparison_plot(all_results_df, output_dir)

    # 2. Optimal threshold comparison
    opt_df = create_optimal_threshold_comparison(optimal_thresholds, output_dir)

    # 3. Metrics heatmap
    heatmap_df = create_metrics_heatmap(all_results_df, optimal_thresholds, output_dir)

    # 4. Clinical impact analysis
    create_clinical_scenario_plot(all_results_df, output_dir)

    # 5. F1/PPV/NPV analysis
    create_f1_ppv_npv_plot(all_results_df, output_dir)

    # 6. Summary table
    summary_df = create_summary_table(all_results_df, optimal_thresholds, output_dir)

    # Print final summary
    print("\n" + "="*70)
    print("EXTENDED THRESHOLD ANALYSIS COMPLETE!")
    print("="*70)

    print("\nKEY FINDINGS:")
    print("-"*50)

    # Best model overall
    best_model = summary_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"   AUROC: {best_model['AUROC']:.4f}")
    print(f"   Optimal Threshold (Youden): {best_model['Optimal_Threshold_Youden']:.2f}")
    print(f"   At this threshold:")
    print(f"     - Sensitivity: {best_model['Sensitivity_at_Optimal']:.1%}")
    print(f"     - Specificity: {best_model['Specificity_at_Optimal']:.1%}")
    print(f"     - PPV: {best_model['PPV_at_Optimal']:.1%}")
    print(f"     - NPV: {best_model['NPV_at_Optimal']:.1%}")
    print(f"     - F1 Score: {best_model['F1_at_Optimal']:.3f}")

    # Threshold consistency across models
    print(f"\nThreshold Consistency:")
    thresholds_youden = [optimal_thresholds[m]['Youden_Index']['threshold'] for m in top_models]
    print(f"   Mean optimal threshold: {np.mean(thresholds_youden):.2f}")
    print(f"   Std: {np.std(thresholds_youden):.2f}")
    print(f"   Range: {min(thresholds_youden):.2f} - {max(thresholds_youden):.2f}")

    # Clinical implication
    print(f"\nCLINICAL IMPLICATION:")
    print(f"   The optimal threshold of ~{np.mean(thresholds_youden):.1f} is BELOW the default 0.5")
    print(f"   This means: Lower the bar for flagging high-risk patients")
    print(f"   Benefit: Catch more potential deaths (higher sensitivity)")
    print(f"   Cost: More false alarms (lower specificity)")

    print(f"\nResults saved in: {output_dir}/")
    print("   - all_threshold_results.csv (raw data)")
    print("   - extended_threshold_summary.csv (summary table)")
    print("   - threshold_comparison_all_models.png")
    print("   - optimal_threshold_comparison.png")
    print("   - metrics_heatmap.png")
    print("   - clinical_impact_analysis.png")
    print("   - f1_ppv_npv_analysis.png")
    print("="*70)


# In[15]:


if __name__ == "__main__":
    main()


# In[16]:


import pandas as pd
perf = pd.read_csv("../results/models/model_performance_summary.csv")
print(perf['Model'].tolist())


# In[17]:


import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("../results/data/train_data.csv")
test = pd.read_csv("../results/data/test_data.csv")

exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']
features = [c for c in test.columns if c not in exclude_cols]

X_train = train[features]
y_train = train['mortality_28day']
X_test = test[features]
y_test = test['mortality_28day']

# Test Complement NB
from sklearn.naive_bayes import ComplementNB

X_train_shift = X_train.copy()
X_test_shift = X_test.copy()
min_val = X_train_shift.min().min()
if min_val < 0:
    shift = -min_val + 0.1
    X_train_shift = X_train_shift + shift
    X_test_shift = X_test_shift + shift

model = ComplementNB()
model.fit(X_train_shift, y_train)
preds = model.predict_proba(X_test_shift)[:, 1]

print(f"Predictions - min: {preds.min():.4f}, max: {preds.max():.4f}, mean: {preds.mean():.4f}, std: {preds.std():.4f}")
print(f"Unique values: {len(np.unique(preds.round(3)))}")
print(f"Distribution:\n{pd.Series(preds).describe()}")


# In[ ]:





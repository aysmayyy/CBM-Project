#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Step 5: Train and Evaluate 10 ML Models
Following Yu et al. methodology exactly:
- 10 models with grid search hyperparameter tuning
- 5-fold cross-validation
- Youden index for optimal threshold
- Bootstrap CI (500 iterations)
- Multiple performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, f1_score)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def load_data():
    """Load preprocessed train and test data"""
    print("Loading preprocessed data...")
    train = pd.read_csv("../results/data/train_data.csv")
    test = pd.read_csv("../results/data/test_data.csv")

    # Separate features and target
    feature_cols = [c for c in train.columns if c not in 
                   ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']]

    X_train = train[feature_cols]
    y_train = train['mortality_28day']
    X_test = test[feature_cols]
    y_test = test['mortality_28day']

    print(f"Training set: {X_train.shape}, Deaths: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"Test set: {X_test.shape}, Deaths: {y_test.sum()} ({y_test.mean():.1%})")

    return X_train, y_train, X_test, y_test, feature_cols


# In[3]:


def apply_yeo_johnson(X_train, X_test):
    """
    Apply Yeo-Johnson transformation for linear models
    Only transform severely skewed features (|skewness| > 1.5)
    """
    print("\nApplying Yeo-Johnson transformation for linear models...")

    # Calculate skewness for each feature
    skewness = X_train.skew()
    skewed_features = skewness[abs(skewness) > 1.5].index.tolist()

    print(f"  Found {len(skewed_features)} severely skewed features")

    if len(skewed_features) > 0:
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)

        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()

        X_train_transformed[skewed_features] = transformer.fit_transform(
            X_train[skewed_features]
        )
        X_test_transformed[skewed_features] = transformer.transform(
            X_test[skewed_features]
        )

        return X_train_transformed, X_test_transformed

    return X_train, X_test


# In[4]:


def get_model_configs():
    """
    Define all 10 models with hyperparameter grids
    """
    configs = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            'needs_transform': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'needs_transform': False
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'needs_transform': False
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 1.0]
            },
            'needs_transform': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            },
            'needs_transform': False
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            'needs_transform': False
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'needs_transform': True
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'needs_transform': True
        },
        'Gaussian Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'needs_transform': True
        },
        'Complement Naive Bayes': {
            'model': ComplementNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'norm': [True, False]
            },
            'needs_transform': True
        }
    }

    return configs


# In[5]:


def find_optimal_threshold_youden(y_true, y_pred_proba):
    """
    Find optimal threshold using Youden's index
    J = sensitivity + specificity - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


# In[6]:


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all performance metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'AUROC': roc_auc_score(y_true, y_pred_proba),
        'AUPRC': average_precision_score(y_true, y_pred_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(y_true, y_pred)
    }

    return metrics


# In[7]:


def bootstrap_ci(y_true, y_pred_proba, optimal_threshold, n_iterations=500):
    """
    Calculate 95% confidence intervals using bootstrap resampling
    """
    print("    Calculating 95% CI with 500 bootstrap iterations...")

    n_samples = len(y_true)
    metrics_bootstrap = []

    np.random.seed(42)

    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true.iloc[indices].values
        y_pred_proba_boot = y_pred_proba[indices]
        y_pred_boot = (y_pred_proba_boot >= optimal_threshold).astype(int)

        # Calculate metrics for this bootstrap sample
        metrics = calculate_metrics(y_true_boot, y_pred_boot, y_pred_proba_boot)
        metrics_bootstrap.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"      Completed {i + 1}/{n_iterations} iterations")

    # Calculate 95% CI (2.5th and 97.5th percentiles)
    ci_results = {}
    df_bootstrap = pd.DataFrame(metrics_bootstrap)

    for metric in df_bootstrap.columns:
        lower = np.percentile(df_bootstrap[metric], 2.5)
        upper = np.percentile(df_bootstrap[metric], 97.5)
        ci_results[metric] = (lower, upper)

    return ci_results


# In[8]:


def train_and_evaluate_model(name, config, X_train, y_train, X_test, y_test,
                             X_train_transformed=None, X_test_transformed=None):
    """
    Train a single model with grid search CV and evaluate
    """
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")

    # Use transformed data for linear models
    if config['needs_transform'] and X_train_transformed is not None:
        X_tr = X_train_transformed
        X_te = X_test_transformed
        print("  Using Yeo-Johnson transformed features")
    else:
        X_tr = X_train
        X_te = X_test
        print("  Using original features (tree-based model)")

    # Special handling for Complement Naive Bayes (needs strictly positive values)
    if name == 'Complement Naive Bayes':
        print("  Shifting to positive values for ComplementNB...")
        # Find minimum value across all features in training set
        min_val = X_tr.min().min()
        if min_val < 0:
            # Shift all values so minimum becomes 0, then add 0.1
            X_tr = X_tr - min_val + 0.1
            X_te = X_te - min_val + 0.1
            print(f"    Shifted by {-min_val + 0.1:.4f} to ensure positive values")

    # Grid search with 5-fold CV
    print("  Performing grid search with 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_tr, y_train)

    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV AUROC: {grid_search.best_score_:.4f}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Predict on test set
    y_pred_proba = best_model.predict_proba(X_te)[:, 1]

    # Find optimal threshold using Youden index
    optimal_threshold = find_optimal_threshold_youden(y_test, y_pred_proba)
    print(f"  Optimal threshold (Youden): {optimal_threshold:.4f}")

    # Make predictions with optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    print(f"\n  Test Set Performance:")
    print(f"    AUROC: {metrics['AUROC']:.4f}")
    print(f"    AUPRC: {metrics['AUPRC']:.4f}")
    print(f"    Accuracy: {metrics['Accuracy']:.4f}")
    print(f"    Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"    Specificity: {metrics['Specificity']:.4f}")
    print(f"    PPV: {metrics['PPV']:.4f}")
    print(f"    NPV: {metrics['NPV']:.4f}")
    print(f"    F1: {metrics['F1']:.4f}")

    # Bootstrap CI
    ci_results = bootstrap_ci(y_test, y_pred_proba, optimal_threshold)

    return {
        'model': best_model,
        'name': name,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'optimal_threshold': optimal_threshold,
        'metrics': metrics,
        'ci': ci_results,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


# In[9]:


def save_results(results, feature_cols):
    """Save all results to files"""
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    try:
        # Create summary dataframe
        summary_data = []

        for result in results:
            row = {
                'Model': result['name'],
                'Optimal_Threshold': result['optimal_threshold'],
                'CV_AUROC': result['best_cv_score']
            }

            # Add metrics
            for metric, value in result['metrics'].items():
                row[metric] = value
                # Add CI
                ci_lower, ci_upper = result['ci'][metric]
                row[f'{metric}_CI_Lower'] = ci_lower
                row[f'{metric}_CI_Upper'] = ci_upper

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('AUROC', ascending=False)

        # Save summary
        output_path = "../results/models/model_performance_summary.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

        # Save detailed results for each model
        for result in results:
            try:
                model_name_clean = result['name'].replace(' ', '_').lower()
                params_df = pd.DataFrame([result['best_params']])
                params_path = f"../results/models/{model_name_clean}_params.csv"
                params_df.to_csv(params_path, index=False)
                print(f"Saved: {params_path}")
            except Exception as e:
                print(f"Warning: Could not save params for {result['name']}: {e}")

        # Print summary table
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print("\nModel Performance (sorted by AUROC):")
        print(summary_df[['Model', 'AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 
                          'Specificity', 'F1']].to_string(index=False))

        # Find best model
        best_model_idx = summary_df['AUROC'].idxmax()
        best_model_name = summary_df.loc[best_model_idx, 'Model']
        best_auroc = summary_df.loc[best_model_idx, 'AUROC']

        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Test AUROC: {best_auroc:.4f}")
        print(f"{'='*70}")

        return summary_df

    except Exception as e:
        print(f"\nERROR saving results: {e}")
        print("Printing results to console instead:")
        for result in results:
            print(f"\n{result['name']}:")
            print(f"  AUROC: {result['metrics']['AUROC']:.4f}")
            print(f"  Best params: {result['best_params']}")
        return None


# In[10]:


def main():
    """Main training pipeline"""
    print("="*70)
    print("ML Model Training Pipeline - Yu et al. Methodology")
    print("="*70)
    print("\nThis will train 10 models with grid search and 5-fold CV")
    print("Estimated time: 30-90 minutes depending on your machine\n")

    # Load data
    X_train, y_train, X_test, y_test, feature_cols = load_data()

    # Apply Yeo-Johnson transformation for linear models
    X_train_transformed, X_test_transformed = apply_yeo_johnson(X_train, X_test)

    # Get model configurations
    configs = get_model_configs()

    # Train all models with error handling
    results = []
    failed_models = []

    for name, config in configs.items():
        try:
            result = train_and_evaluate_model(
                name, config, 
                X_train, y_train, 
                X_test, y_test,
                X_train_transformed, 
                X_test_transformed
            )
            results.append(result)
        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"ERROR: {name} failed to train")
            print(f"Error message: {e}")
            print(f"{'!'*70}\n")
            failed_models.append(name)
            continue

    # Save results even if some models failed
    if len(results) > 0:
        summary_df = save_results(results, feature_cols)
    else:
        print("\nERROR: No models completed successfully!")
        return

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    if failed_models:
        print(f"\nWarning: {len(failed_models)} model(s) failed to train:")
        for model in failed_models:
            print(f"  - {model}")

    print(f"\nSuccessfully trained: {len(results)} models")
    print("\nNext steps:")
    print("1. Review model_performance_summary.csv")
    print("2. Run SHAP analysis on best model")
    print("3. Perform decision curve analysis")
    print("4. Create visualizations")
    print("="*70)


# In[11]:


if __name__ == "__main__":
    main()


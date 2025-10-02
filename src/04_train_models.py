"""
Step 4: Train All 10 Machine Learning Models
Reproduces the full model comparison from the paper
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, roc_curve,
                            precision_recall_curve, auc)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load data and prepare for modeling"""
    
    print("Loading data...")
    df = pd.read_csv("../results/complete_ml_features.csv")
    
    print(f"Initial dataset: {df.shape}")
    print(f"Mortality rate: {df['mortality_28day'].mean():.1%}")
    
    # Encode categorical variables first
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    # Select features for modeling
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 
                    'last_careunit', 'intime', 'outtime', 'mortality_28day']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['mortality_28day'].copy()
    
    # Convert all columns to numeric, coerce errors to NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Handle missing values
    print("\nHandling missing values...")
    missing_pct = (X.isnull().sum() / len(X) * 100).round(1)
    
    # Remove features with >20% missing
    high_missing = missing_pct[missing_pct > 20].index
    if len(high_missing) > 0:
        print(f"Removing {len(high_missing)} features with >20% missing")
        X = X.drop(columns=high_missing)
    
    # Impute remaining missing values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"Final features: {X_imputed.shape[1]}")
    
    return X_imputed, y, X_imputed.columns.tolist()

def create_models():
    """Define all 10 models with hyperparameter grids"""
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42, algorithm='SAMME'),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear']
            }
        },
        'Gaussian Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        },
        'Complement Naive Bayes': {
            'model': ComplementNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0]
            }
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(100,), (100, 50)],
                'alpha': [0.0001, 0.001],
                'learning_rate_init': [0.001, 0.01]
            }
        }
    }
    
    return models

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all performance metrics"""
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': tn / (tn + fp),
        'ppv': precision_score(y_true, y_pred),
        'npv': tn / (tn + fn),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics['auprc'] = auc(recall, precision)
    
    return metrics

def train_and_evaluate_models(X, y):
    """Train all models with cross-validation"""
    
    print("\n=== Training Models ===\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Test mortality rate: {y_test.mean():.1%}\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get models
    models = create_models()
    
    results = []
    trained_models = {}
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model_info in models.items():
        print(f"Training {model_name}...")
        
        try:
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predictions on test set
            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            
            # Store results
            result = {
                'Model': model_name,
                'AUC': metrics['auc'],
                'AUPRC': metrics['auprc'],
                'Accuracy': metrics['accuracy'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'PPV': metrics['ppv'],
                'NPV': metrics['npv'],
                'F1': metrics['f1'],
                'Best_Params': str(grid_search.best_params_)
            }
            
            results.append(result)
            trained_models[model_name] = best_model
            
            print(f"  AUC: {metrics['auc']:.3f}, AUPRC: {metrics['auprc']:.3f}")
            
        except Exception as e:
            print(f"  Error training {model_name}: {e}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC', ascending=False)
    
    return results_df, trained_models, scaler, X_test_scaled, y_test

def save_results(results_df):
    """Save model performance results"""
    
    output_file = "../results/model_performance.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n=== Model Performance Summary ===\n")
    print(results_df[['Model', 'AUC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']].to_string(index=False))
    
    print(f"\n✓ Results saved to: {output_file}")

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("ML MODEL TRAINING - MIMIC-IV Immunocompromised Patients")
    print("=" * 60)
    
    # Load and preprocess
    X, y, feature_names = load_and_preprocess_data()
    
    # Train models
    results_df, trained_models, scaler, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Save results
    save_results(results_df)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
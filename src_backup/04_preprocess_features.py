#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install statsmodels')


# In[2]:


"""
Step 4: Preprocess Features - Correct Methodology (No Data Leakage)
- Exclude variables with >20% missing (before split)
- Split into train/test 
- KNN imputation fitted on TRAIN only
- Univariate logistic regression on TRAIN only (remove P>0.05)
- Correlation analysis on TRAIN only (remove highly correlated features)
- One-hot encoding for categorical variables
- Apply same transformations to test set
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, chi2_contingency
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[3]:


def load_features():
    """Load extracted features"""
    features = pd.read_csv("../results/data/complete_ml_features.csv")
    print(f"Loaded {len(features)} patients with {features.shape[1]} features")

    # Drop problematic string/datetime columns that shouldn't be in features
    cols_to_drop = ['first_careunit', 'last_careunit', 'intime', 'outtime']
    cols_to_drop = [c for c in cols_to_drop if c in features.columns]

    if cols_to_drop:
        print(f"\nDropping non-feature columns: {cols_to_drop}")
        features = features.drop(columns=cols_to_drop)

    # Fix duplicate gender columns if they exist
    if 'gender_x' in features.columns and 'gender_y' in features.columns:
        print("Fixing duplicate gender columns...")
        features['gender'] = features['gender_x'].fillna(features['gender_y'])
        features = features.drop(columns=['gender_x', 'gender_y'])
    elif 'gender_x' in features.columns:
        features = features.rename(columns={'gender_x': 'gender'})
    elif 'gender_y' in features.columns:
        features = features.rename(columns={'gender_y': 'gender'})

    print(f"After cleaning: {features.shape[1]} features")

    return features


# In[4]:


def exclude_high_missing(data, threshold=0.20):
    """
    Step 1: Exclude variables with >20% missing values
    This is a simple rule that doesn't leak information, so we do it before splitting
    """
    print("\n" + "="*70)
    print("STEP 1: Excluding variables with >20% missing values")
    print("="*70)

    # Don't check these columns for missing
    protected_cols = ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']

    missing_pct = data.isnull().sum() / len(data)
    high_missing = missing_pct[missing_pct > threshold]

    # Exclude protected columns from removal
    cols_to_remove = [col for col in high_missing.index if col not in protected_cols]

    print(f"\nVariables with >{threshold*100}% missing:")
    for col in cols_to_remove:
        print(f"  {col}: {missing_pct[col]*100:.1f}%")

    print(f"\nRemoving {len(cols_to_remove)} variables")

    data_filtered = data.drop(columns=cols_to_remove)

    print(f"Remaining features: {data_filtered.shape[1]}")

    return data_filtered, cols_to_remove


# In[5]:


def split_train_test(data, test_size=0.2, random_state=42):
    """
    Step 2: Split into train/test BEFORE any learning-based preprocessing
    This prevents data leakage
    """
    print("\n" + "="*70)
    print("STEP 2: Splitting into train/test sets")
    print("="*70)

    # Separate IDs, features, and outcome
    id_cols = ['subject_id', 'hadm_id', 'stay_id']
    outcome_col = 'mortality_28day'

    X = data.drop(columns=id_cols + [outcome_col])
    y = data[outcome_col]
    ids = data[id_cols]

    # Stratified split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples ({y_train.mean():.1%} mortality)")
    print(f"Test set: {len(X_test)} samples ({y_test.mean():.1%} mortality)")

    return X_train, X_test, y_train, y_test, ids_train, ids_test


# In[6]:


def identify_column_types(X_train):
    """Identify categorical vs numeric columns"""

    categorical_cols = ['gender', 'ethnicity', 'myocardial_infarction', 
                       'congestive_heart_failure', 'peripheral_vascular_disease',
                       'cerebrovascular_disease', 'dementia', 'chronic_pulmonary_disease',
                       'rheumatic_disease', 'peptic_ulcer_disease', 'mild_liver_disease',
                       'diabetes_without_cc', 'diabetes_with_cc', 'paraplegia',
                       'renal_disease', 'malignant_cancer', 'severe_liver_disease',
                       'metastatic_solid_tumor', 'aids', 'mechanical_ventilation']

    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    return numeric_cols, categorical_cols


# In[7]:


def impute_missing_values(X_train, X_test, numeric_cols, categorical_cols):
    """
    Step 3: Impute missing values
    Fit imputer on TRAIN only, then transform both train and test
    """
    print("\n" + "="*70)
    print("STEP 3: Imputing missing values (KNN for numeric, mode for categorical)")
    print("="*70)

    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    # Impute numeric features with KNN (fit on train only)
    if numeric_cols:
        print("\nImputing numeric features...")
        imputer_numeric = KNNImputer(n_neighbors=5)
        X_train_imputed[numeric_cols] = imputer_numeric.fit_transform(X_train[numeric_cols])
        X_test_imputed[numeric_cols] = imputer_numeric.transform(X_test[numeric_cols])
        print("  Numeric imputation complete")

    # Impute categorical features with mode (fit on train only)
    if categorical_cols:
        print("\nImputing categorical features with mode...")
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        X_train_imputed[categorical_cols] = imputer_categorical.fit_transform(X_train[categorical_cols])
        X_test_imputed[categorical_cols] = imputer_categorical.transform(X_test[categorical_cols])
        print("  Categorical imputation complete")

    # Check for any remaining missing values
    remaining_train = X_train_imputed.isnull().sum().sum()
    remaining_test = X_test_imputed.isnull().sum().sum()
    print(f"\nRemaining missing values - Train: {remaining_train}, Test: {remaining_test}")

    return X_train_imputed, X_test_imputed


# In[8]:


def univariate_feature_selection(X_train, y_train, numeric_cols, categorical_cols, p_threshold=0.05):
    """
    Step 4: Univariate logistic regression on TRAIN set only
    Returns list of features to keep
    """
    print("\n" + "="*70)
    print("STEP 4: Univariate feature selection (P>0.05 removal, TRAIN only)")
    print("="*70)

    features_to_test = numeric_cols + categorical_cols

    print(f"\nTesting {len(features_to_test)} features...")

    p_values = {}

    for feature in features_to_test:
        try:
            X = X_train[[feature]].copy()

            # Add constant for statsmodels
            X_with_const = sm.add_constant(X)

            # Fit logistic regression
            model = sm.Logit(y_train, X_with_const)
            result = model.fit(disp=0)

            # Get p-value for the feature (not the constant)
            p_value = result.pvalues[1] if len(result.pvalues) > 1 else 1.0
            p_values[feature] = p_value

        except Exception as e:
            # If fitting fails, mark as non-significant
            p_values[feature] = 1.0

    # Filter features based on p-value threshold
    significant_features = [f for f, p in p_values.items() if p <= p_threshold]
    removed_features = [f for f, p in p_values.items() if p > p_threshold]

    print(f"\nSignificant features (P≤{p_threshold}): {len(significant_features)}")
    print(f"Removed features (P>{p_threshold}): {len(removed_features)}")

    if removed_features:
        print("\nRemoved features (top 10):")
        removed_sorted = sorted([(f, p_values[f]) for f in removed_features], 
                               key=lambda x: x[1], reverse=True)
        for feat, pval in removed_sorted[:10]:
            print(f"  {feat}: P={pval:.4f}")
        if len(removed_features) > 10:
            print(f"  ... and {len(removed_features)-10} more")

    return significant_features


# In[9]:


def calculate_cramers_v(x, y):
    """Calculate Cramer's V for categorical-categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


# In[10]:


def calculate_correlation_ratio(categories, values):
    """Calculate correlation ratio (eta) for categorical-numeric association"""
    try:
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)

        for i in range(cat_num):
            cat_measures = values[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0

        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))

        if denominator == 0:
            return 0
        else:
            return np.sqrt(numerator / denominator)
    except:
        return 0


# In[11]:


def remove_correlated_features(X_train, y_train, features_to_keep, categorical_cols, threshold=0.75):
    """
    Step 5: Remove highly correlated features using TRAIN set only
    Uses Spearman for numeric, Cramer's V for categorical, correlation ratio for mixed
    """
    print("\n" + "="*70)
    print("STEP 5: Removing highly correlated features (threshold=0.75, TRAIN only)")
    print("="*70)

    numeric_features = [f for f in features_to_keep if f not in categorical_cols]
    categorical_features = [f for f in features_to_keep if f in categorical_cols]

    print(f"\nAnalyzing correlations among {len(features_to_keep)} features...")
    print(f"  Numeric: {len(numeric_features)}")
    print(f"  Categorical: {len(categorical_features)}")

    features_to_remove = set()

    # Check numeric-numeric correlations (Spearman)
    print("\nChecking numeric-numeric correlations (Spearman)...")
    if len(numeric_features) > 1:
        for i, feat1 in enumerate(numeric_features):
            if feat1 in features_to_remove:
                continue
            for feat2 in numeric_features[i+1:]:
                if feat2 in features_to_remove:
                    continue

                try:
                    corr, _ = spearmanr(X_train[feat1], X_train[feat2], nan_policy='omit')

                    if abs(corr) > threshold:
                        # Keep the feature with stronger association to outcome
                        corr1, _ = spearmanr(X_train[feat1], y_train, nan_policy='omit')
                        corr2, _ = spearmanr(X_train[feat2], y_train, nan_policy='omit')

                        if abs(corr1) >= abs(corr2):
                            features_to_remove.add(feat2)
                            print(f"  Removing {feat2} (corr={corr:.3f} with {feat1})")
                        else:
                            features_to_remove.add(feat1)
                            print(f"  Removing {feat1} (corr={corr:.3f} with {feat2})")
                            break
                except:
                    continue

    # Check categorical-categorical correlations (Cramer's V)
    print("\nChecking categorical-categorical correlations (Cramer's V)...")
    if len(categorical_features) > 1:
        for i, feat1 in enumerate(categorical_features):
            if feat1 in features_to_remove:
                continue
            for feat2 in categorical_features[i+1:]:
                if feat2 in features_to_remove:
                    continue

                try:
                    v = calculate_cramers_v(X_train[feat1], X_train[feat2])

                    if v > threshold:
                        # Keep the feature with stronger association to outcome
                        v1 = calculate_cramers_v(X_train[feat1], y_train)
                        v2 = calculate_cramers_v(X_train[feat2], y_train)

                        if v1 >= v2:
                            features_to_remove.add(feat2)
                            print(f"  Removing {feat2} (Cramer's V={v:.3f} with {feat1})")
                        else:
                            features_to_remove.add(feat1)
                            print(f"  Removing {feat1} (Cramer's V={v:.3f} with {feat2})")
                            break
                except:
                    continue

    # Check mixed correlations (correlation ratio)
    print("\nChecking categorical-numeric correlations (Correlation Ratio)...")
    for cat_feat in categorical_features:
        if cat_feat in features_to_remove:
            continue
        for num_feat in numeric_features:
            if num_feat in features_to_remove:
                continue

            try:
                eta = calculate_correlation_ratio(X_train[cat_feat], X_train[num_feat])

                if eta > threshold:
                    # Keep numeric over categorical (generally more informative)
                    features_to_remove.add(cat_feat)
                    print(f"  Removing {cat_feat} (eta={eta:.3f} with {num_feat})")
                    break
            except:
                continue

    final_features = [f for f in features_to_keep if f not in features_to_remove]

    print(f"\nRemoved {len(features_to_remove)} correlated features")
    print(f"Remaining features: {len(final_features)}")

    return final_features


# In[12]:


def one_hot_encode_categorical(X_train, X_test, categorical_cols):
    """
    Step 6: One-hot encode categorical variables
    Fit on train, apply same encoding to test
    """
    print("\n" + "="*70)
    print("STEP 6: One-hot encoding categorical variables")
    print("="*70)

    categorical_cols_present = [c for c in categorical_cols if c in X_train.columns]

    print(f"\nEncoding {len(categorical_cols_present)} categorical variables...")

    # One-hot encode train
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols_present, drop_first=True)

    # One-hot encode test (align columns with train)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols_present, drop_first=True)

    # Ensure test has same columns as train
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0

    # Remove extra columns in test
    extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
    X_test_encoded = X_test_encoded.drop(columns=list(extra_cols))

    # Reorder test columns to match train
    X_test_encoded = X_test_encoded[X_train_encoded.columns]

    new_cols = len(X_train_encoded.columns) - len(X_train.columns)
    print(f"Created {new_cols} new binary columns")
    print(f"Total features after encoding: {X_train_encoded.shape[1]}")

    return X_train_encoded, X_test_encoded


# In[13]:


def save_preprocessed_data(X_train, X_test, y_train, y_test, ids_train, ids_test):
    """Save the final preprocessed datasets"""
    print("\n" + "="*70)
    print("STEP 7: Saving preprocessed data")
    print("="*70)

    # Combine features with IDs and outcome
    train_data = pd.concat([ids_train.reset_index(drop=True), 
                           X_train.reset_index(drop=True), 
                           y_train.reset_index(drop=True)], axis=1)

    test_data = pd.concat([ids_test.reset_index(drop=True), 
                          X_test.reset_index(drop=True), 
                          y_test.reset_index(drop=True)], axis=1)

    # Save
    train_file = "../results/data/train_data.csv"
    test_file = "../results/data/test_data.csv"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"\nSaved training data: {train_file}")
    print(f"  Shape: {train_data.shape}")
    print(f"  Mortality: {y_train.mean():.1%}")

    print(f"\nSaved test data: {test_file}")
    print(f"  Shape: {test_data.shape}")
    print(f"  Mortality: {y_test.mean():.1%}")

    return train_data, test_data


# In[14]:


def main():
    """Main preprocessing pipeline - NO DATA LEAKAGE"""

    print("="*70)
    print("FEATURE PREPROCESSING PIPELINE (No Data Leakage)")
    print("="*70)

    # Load data
    data = load_features()

    # Step 1: Exclude high missing (simple rule, no leakage)
    data, removed_high_missing = exclude_high_missing(data, threshold=0.20)

    # Step 2: Split BEFORE any learning
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_train_test(
        data, test_size=0.2, random_state=42
    )

    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X_train)

    # Step 3: Impute (fit on train only)
    X_train, X_test = impute_missing_values(
        X_train, X_test, numeric_cols, categorical_cols
    )

    # Step 4: Univariate selection (on train only)
    significant_features = univariate_feature_selection(
        X_train, y_train, numeric_cols, categorical_cols, p_threshold=0.05
    )

    # Apply to both train and test
    X_train = X_train[significant_features]
    X_test = X_test[significant_features]

    # Update column lists
    categorical_cols = [c for c in categorical_cols if c in significant_features]

    # Step 5: Remove correlated (on train only)
    final_features = remove_correlated_features(
        X_train, y_train, significant_features, categorical_cols, threshold=0.75
    )

    # Apply to both train and test
    X_train = X_train[final_features]
    X_test = X_test[final_features]

    # Update categorical cols
    categorical_cols = [c for c in categorical_cols if c in final_features]

    # Step 6: One-hot encode (fit on train, apply to test)
    X_train, X_test = one_hot_encode_categorical(X_train, X_test, categorical_cols)

    # Step 7: Save
    train_data, test_data = save_preprocessed_data(
        X_train, X_test, y_train, y_test, ids_train, ids_test
    )

    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE - NO DATA LEAKAGE!")
    print("="*70)
    print(f"\nOriginal features: {len([c for c in load_features().columns if c not in ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']])}")
    print(f"After removing >20% missing: {len(data.columns) - 4}")
    print(f"After univariate selection: {len(significant_features)}")
    print(f"After correlation removal: {len(final_features)}")
    print(f"After one-hot encoding: {X_train.shape[1]}")

    print(f"\nFinal datasets:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    print(f"\n{'='*70}")
    print("Next step: Model training!")
    print("Use train_data.csv for model development")
    print("Use test_data.csv for final evaluation only")
    print("="*70)


# In[15]:


if __name__ == "__main__":
    main()



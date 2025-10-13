"""
Step 5: Feature Selection - EXACT Paper Replication
Implements all methods exactly as described in the paper
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from scipy.stats import spearmanr
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the complete feature dataset"""
    
    print("Loading complete features...")
    df = pd.read_csv("../results/complete_ml_features.csv")
    
    print(f"Initial dataset: {df.shape}")
    print(f"Mortality rate: {df['mortality_28day'].mean():.1%}\n")
    
    return df

def prepare_features(df):
    """Separate features from outcome"""
    
    print("Preparing features...")
    
    # Define columns to exclude
    exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 
                    'last_careunit', 'intime', 'outtime', 'mortality_28day']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['mortality_28day'].copy()
    
    print(f"Starting features: {X.shape[1]}")
    
    return X, y

def identify_categorical_features(X):
    """Identify categorical vs numeric features"""
    
    categorical = []
    numeric = []
    
    for col in X.columns:
        # Check if column has few unique values or is object type
        unique_vals = X[col].nunique()
        if unique_vals <= 10 or X[col].dtype == 'object':
            categorical.append(col)
        else:
            numeric.append(col)
    
    return categorical, numeric

def one_hot_encode_categoricals(X, categorical_features):
    """One-hot encode categorical variables"""
    
    if len(categorical_features) == 0:
        return X
    
    print(f"\nOne-hot encoding {len(categorical_features)} categorical features...")
    
    # Special handling for gender
    if 'gender' in X.columns:
        X['gender'] = X['gender'].map({'M': 1, 'F': 0})
        if 'gender' in categorical_features:
            categorical_features.remove('gender')
    
    # One-hot encode remaining categoricals
    if len(categorical_features) > 0:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    return X

def remove_high_missing(X, threshold=0.20):
    """Remove variables with >20% missing values"""
    
    print(f"\n=== Step 1: Remove features with >{threshold*100}% missing ===")
    
    missing_pct = X.isnull().sum() / len(X)
    high_missing = missing_pct[missing_pct > threshold].index.tolist()
    
    if len(high_missing) > 0:
        print(f"Removing {len(high_missing)} features with >20% missing:")
        for col in high_missing:
            print(f"  - {col}: {missing_pct[col]*100:.1f}%")
        X = X.drop(columns=high_missing)
    else:
        print("No features with >20% missing")
    
    print(f"Remaining features: {X.shape[1]}")
    
    return X

def impute_missing(X):
    """Nearest neighbor imputation for remaining missing values"""
    
    print("\n=== Step 2: Nearest Neighbor Imputation ===")
    
    # Convert all to numeric first
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    missing_before = X.isnull().sum().sum()
    print(f"Total missing values: {missing_before}")
    
    if missing_before > 0:
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print("Imputation complete")
    else:
        X_imputed = X.copy()
        print("No missing values to impute")
    
    print(f"Features after imputation: {X_imputed.shape[1]}")
    
    return X_imputed

def univariate_logistic_regression(X, y, p_threshold=0.05):
    """
    Univariate logistic regression with proper p-values using statsmodels
    Eliminates features with P > 0.05
    """
    
    print(f"\n=== Step 3: Univariate Logistic Regression (P < {p_threshold}) ===")
    print("Calculating p-values for each feature...")
    
    selected_features = []
    p_values = {}
    
    for col in X.columns:
        try:
            # Prepare data for statsmodels (add constant)
            X_col = sm.add_constant(X[[col]])
            
            # Fit logistic regression
            model = sm.Logit(y, X_col)
            result = model.fit(disp=0, maxiter=100)
            
            # Get p-value for the feature (not the constant)
            p_value = result.pvalues[col]
            p_values[col] = p_value
            
            if p_value < p_threshold:
                selected_features.append(col)
                
        except Exception as e:
            # If model fails to converge, assume not significant
            p_values[col] = 1.0
            continue
    
    print(f"\nFeatures with P < {p_threshold}: {len(selected_features)}")
    print(f"Features removed: {X.shape[1] - len(selected_features)}")
    
    # Show most significant features
    p_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-value'])
    p_df = p_df.sort_values('P-value')
    
    print("\nTop 20 most significant features:")
    print(p_df.head(20).to_string(index=False))
    
    print(f"\nTotal features retained: {len(selected_features)}")
    
    return X[selected_features], p_values

def cramers_v(x, y):
    """Calculate Cramer's V for categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def correlation_ratio(categories, values):
    """Calculate correlation ratio for mixed (categorical-numeric) pairs"""
    categories = np.array(categories)
    values = np.array(values)
    
    # Overall mean
    mean_total = np.mean(values)
    
    # Group means
    categories_unique = np.unique(categories)
    ss_between = 0
    
    for cat in categories_unique:
        cat_values = values[categories == cat]
        cat_mean = np.mean(cat_values)
        ss_between += len(cat_values) * (cat_mean - mean_total) ** 2
    
    # Total sum of squares
    ss_total = np.sum((values - mean_total) ** 2)
    
    if ss_total == 0:
        return 0
    
    return np.sqrt(ss_between / ss_total)

def reduce_multicollinearity(X, y, p_values, corr_threshold=0.75):
    """
    Reduce multicollinearity using appropriate correlation measures:
    - Spearman for numeric-numeric pairs
    - Cramer's V for categorical-categorical pairs  
    - Correlation ratio for mixed pairs
    Keep feature with lower p-value from univariate regression
    """
    
    print(f"\n=== Step 4: Correlation Analysis (threshold = {corr_threshold}) ===")
    
    # Identify numeric vs categorical features
    categorical_features = []
    numeric_features = []
    
    for col in X.columns:
        unique_count = X[col].nunique()
        if unique_count <= 10:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    features_to_remove = set()
    
    # Calculate correlations for all pairs
    for i, col1 in enumerate(X.columns):
        if col1 in features_to_remove:
            continue
            
        for col2 in X.columns[i+1:]:
            if col2 in features_to_remove:
                continue
            
            # Determine correlation type
            is_col1_cat = col1 in categorical_features
            is_col2_cat = col2 in categorical_features
            
            try:
                if not is_col1_cat and not is_col2_cat:
                    # Both numeric: Spearman correlation
                    corr, _ = spearmanr(X[col1], X[col2])
                    corr = abs(corr)
                elif is_col1_cat and is_col2_cat:
                    # Both categorical: Cramer's V
                    corr = cramers_v(X[col1], X[col2])
                else:
                    # Mixed: Correlation ratio
                    if is_col1_cat:
                        corr = correlation_ratio(X[col1], X[col2])
                    else:
                        corr = correlation_ratio(X[col2], X[col1])
                
                # If correlation exceeds threshold, remove feature with higher p-value
                if corr > corr_threshold:
                    p1 = p_values.get(col1, 1.0)
                    p2 = p_values.get(col2, 1.0)
                    
                    if p1 > p2:
                        features_to_remove.add(col1)
                        print(f"  Removing {col1} (p={p1:.4f}) - correlated with {col2} (p={p2:.4f}), corr={corr:.3f}")
                    else:
                        features_to_remove.add(col2)
                        print(f"  Removing {col2} (p={p2:.4f}) - correlated with {col1} (p={p1:.4f}), corr={corr:.3f}")
                        
            except Exception as e:
                continue
    
    # Remove correlated features
    remaining_features = [col for col in X.columns if col not in features_to_remove]
    X_reduced = X[remaining_features]
    
    print(f"\nFeatures removed due to multicollinearity: {len(features_to_remove)}")
    print(f"Final features: {len(remaining_features)}")
    
    return X_reduced, remaining_features

def identify_skewed_features(X, threshold=1.5):
    """Identify severely skewed features (|skewness| > 1.5)"""
    
    print("\n=== Identifying Skewed Features ===")
    
    skewness = X.skew()
    severely_skewed = skewness[abs(skewness) > threshold]
    
    print(f"Severely skewed features (|skew| > {threshold}): {len(severely_skewed)}")
    
    if len(severely_skewed) > 0:
        print("\nMost skewed features:")
        print(severely_skewed.abs().sort_values(ascending=False).head(10))
    
    return severely_skewed.index.tolist()

def main():
    """Main feature selection pipeline - exact paper replication"""
    
    print("=" * 70)
    print("FEATURE SELECTION - EXACT PAPER REPLICATION")
    print("=" * 70)
    
    # Load data
    df = load_data()
    X, y = prepare_features(df)
    
    # Identify categorical features before encoding
    categorical_features, numeric_features = identify_categorical_features(X)
    print(f"\nInitial categorical features: {len(categorical_features)}")
    print(f"Initial numeric features: {len(numeric_features)}")
    
    # One-hot encode categoricals
    X = one_hot_encode_categoricals(X, categorical_features)
    
    # Step 1: Remove high missing (>20%)
    X = remove_high_missing(X, threshold=0.20)
    
    # Step 2: Nearest neighbor imputation
    X = impute_missing(X)
    
    # Step 3: Univariate logistic regression (P < 0.05)
    X, p_values = univariate_logistic_regression(X, y, p_threshold=0.05)
    
    # Step 4: Correlation analysis with proper measures
    X, final_features = reduce_multicollinearity(X, y, p_values, corr_threshold=0.75)
    
    # Identify skewed features for preprocessing guidance
    skewed_features = identify_skewed_features(X, threshold=1.5)
    
    # Combine with outcome
    final_df = X.copy()
    final_df['mortality_28day'] = y
    
    # Save results
    output_file = "../results/selected_features.csv"
    final_df.to_csv(output_file, index=False)
    
    # Save feature metadata
    feature_info = pd.DataFrame({
        'feature': final_features,
        'p_value': [p_values.get(f, np.nan) for f in final_features],
        'skewed': [f in skewed_features for f in final_features]
    })
    feature_info = feature_info.sort_values('p_value')
    feature_info.to_csv("../results/feature_list.csv", index=False)
    
    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Final features: {len(final_features)}")
    print(f"Severely skewed features: {len(skewed_features)}")
    print(f"\nFiles saved:")
    print(f"  - {output_file}")
    print(f"  - ../results/feature_list.csv")

if __name__ == "__main__":
    main()
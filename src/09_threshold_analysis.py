"""
Threshold Analysis - Finding the Best Decision Threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier

# Load data
print("Loading data...")
train = pd.read_csv("../results/train_data.csv")
test = pd.read_csv("../results/test_data.csv")

# Get features (remove ID columns and outcome)
features = [c for c in test.columns if c not in 
           ['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']]

X_train = train[features]
y_train = train['mortality_28day']
X_test = test[features]
y_test = test['mortality_28day']

# Train model
print("Training model...")
model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# Get predictions
predictions = model.predict_proba(X_test)[:, 1]

print(f"\nTest patients: {len(y_test)}")
print(f"Actual deaths: {y_test.mean():.1%}")
print(f"Model AUROC: {roc_auc_score(y_test, predictions):.4f}")

# Test different thresholds
print("\n" + "="*60)
print("Testing different thresholds...")
print("="*60)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
results = []

for threshold in thresholds:
    # Classify patients
    predicted = (predictions >= threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    results.append({
        'Threshold': threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'True_Positive': tp,
        'False_Positive': fp
    })
    
    print(f"Threshold {threshold:.1f}: Sens={sensitivity:.3f}, Spec={specificity:.3f}")

# Save
df = pd.DataFrame(results)
df.to_csv("../results/threshold_analysis/threshold_analysis.csv", index=False)
print("\nSaved results")

# Simple plot
plt.figure(figsize=(10, 6))
plt.plot(df['Threshold'], df['Sensitivity'], 'o-', label='Sensitivity', linewidth=2)
plt.plot(df['Threshold'], df['Specificity'], 's-', label='Specificity', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Performance')
plt.title('Sensitivity vs Specificity at Different Thresholds')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("../results/threshold_analysis/threshold_analysis.png", dpi=300)
print("Saved plot")

# Find best threshold
df['Total'] = df['Sensitivity'] + df['Specificity']
best = df.loc[df['Total'].idxmax()]

print("\n" + "="*60)
print("Best threshold:", best['Threshold'])
print(f"Sensitivity: {best['Sensitivity']:.3f}")
print(f"Specificity: {best['Specificity']:.3f}")
print("="*60)

print("\nDone!")
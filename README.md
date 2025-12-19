# Reproducibility Study: Predicting 28-Day ICU Mortality in Immunocompromised Patients

A reproducibility study and extension of Yu et al. (2025) "Explainable machine learning model for prediction of 28-day all-cause mortality in immunocompromised patients in the intensive care unit."

## Overview

This project reproduces the machine learning pipeline from Yu et al. using MIMIC-IV v3.1 and extends it with threshold optimization analysis to provide actionable clinical guidance.

**Key Findings:**
- Successfully reproduced 10 ML models with AUROC 0.73-0.87
- LightGBM achieved best performance (AUROC: 0.867)
- **Extension:** Optimal decision threshold is 0.15-0.20, not the default 0.5
- At optimal threshold: 78% sensitivity, 77% specificity, 95% NPV

## Repository Structure

```
CBM-Project/
├── docs/                          # Original paper and supplementary materials
├── notebooks/                     # Analysis notebooks (run in order)
│   ├── 01_explore_data.ipynb           # Initial data exploration
│   ├── 02_identify_patients.ipynb      # Cohort selection using ICD codes
│   ├── 03_extract_features.ipynb       # Feature extraction from MIMIC-IV
│   ├── 04_preprocess_features.ipynb    # Missing value handling, feature selection
│   ├── 05_train_ml_models.ipynb        # Train all 10 ML models
│   ├── 06_cox_regression_analysis.ipynb# Cox proportional hazards analysis
│   ├── 07_shap_analysis.ipynb          # SHAP feature importance
│   ├── 08_clinical_utility_analysis.ipynb # Decision curves, calibration
│   ├── 09_generate_figures.ipynb       # Publication-ready figures
│   ├── 10_threshold_analysis.ipynb     # EXTENSION: Threshold optimization
│   ├── baseline_characteristics_table.ipynb
│   ├── check_cohort.ipynb
│   ├── check_nonnumeric_features.ipynb
│   ├── check_urine_items.ipynb
│   └── ...
├── results/                       # Output files
│   ├── calibration_analysis/      # Calibration curves and metrics
│   ├── clinical_utility/          # Decision curve analysis
│   ├── cox_analysis/              # Cox regression results
│   ├── data/                      # Processed datasets
│   ├── extended_threshold_analysis/ # EXTENSION results
│   ├── figures/                   # All generated figures
│   ├── models/                    # Saved model files
│   ├── shap_analysis/             # SHAP plots and values
│   ├── tables/                    # Summary tables (CSV)
│   └── threshold_analysis/        # Threshold analysis outputs
├── src_backup/                    # Backup scripts
├── requirements.txt               # Python dependencies
└── .gitignore
```

## How to Run

### Prerequisites
1. Access to MIMIC-IV v3.1 (requires PhysioNet credentialing)
2. Python 3.10+

### Installation
```bash
git clone https://github.com/aysmayyy/CBM-Project.git
cd CBM-Project
pip install -r requirements.txt
```

### Run Notebooks
Run notebooks in numerical order (01 → 10). Each notebook saves outputs to `results/`.

**Note:** Notebooks 01-04 require MIMIC-IV data access (../data/mimic-iv-3.1). Notebooks 05-10 can run on the processed data in `results/data/`.

## Extension: Threshold Optimization

The original paper reports AUROC but doesn't specify decision thresholds for clinical use. Our extension (`10_threshold_analysis.ipynb`) addresses this gap:

- Tests thresholds 0.05-0.95 across all 10 models
- Finds optimal threshold using Youden Index
- **Result:** Optimal threshold = 0.15-0.20 (not 0.5)
- Clinical recommendation: Flag patients with ≥15-20% predicted mortality risk

## Results Summary

| Model | AUROC | Optimal Threshold | Sensitivity | Specificity | NPV |
|-------|-------|-------------------|-------------|-------------|-----|
| LightGBM | 0.867 | 0.16 | 78.4% | 78.6% | 94.6% |
| XGBoost | 0.864 | 0.17 | 75.7% | 80.4% | 94.1% |
| Gradient Boosting | 0.861 | 0.18 | 75.1% | 80.7% | 94.0% |
| Random Forest | 0.856 | 0.22 | 73.6% | 80.4% | 93.7% |
| MLP | 0.849 | 0.17 | 76.8% | 77.3% | 94.2% |

## Reference

Yu Z, Fang L, Ding Y. Explainable machine learning model for prediction of 28-day all-cause mortality in immunocompromised patients in the intensive care unit. Eur J Med Res. 2025;30(1):358.

## Author

Asma Nawaz - Computational Biomedicine, Fall 2025

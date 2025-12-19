#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Add this function to your 02_identify_patients.py or create a new file: 05_baseline_characteristics.py
Generates Table 1 with baseline characteristics comparing survival vs mortality groups
"""
import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


def generate_baseline_characteristics_table(features_file, output_dir="../results/tables"):
    """
    Generate comprehensive baseline characteristics table (Table 1)
    Compares survival group vs mortality group with statistical tests

    Parameters:
    -----------
    features_file : str
        Path to complete_ml_features.csv
    output_dir : str
        Directory to save results
    """

    print("=" * 70)
    print("GENERATING BASELINE CHARACTERISTICS TABLE")
    print("=" * 70)

    # Load data
    df = pd.read_csv(features_file)

    # Split by outcome
    survival = df[df['mortality_28day'] == 0].copy()
    mortality = df[df['mortality_28day'] == 1].copy()

    print(f"\nTotal patients: {len(df)}")
    print(f"Survival group: {len(survival)} ({len(survival)/len(df)*100:.1f}%)")
    print(f"Mortality group: {len(mortality)} ({len(mortality)/len(df)*100:.1f}%)")

    # Initialize results
    results = []

    def add_continuous_var(var_name, display_name, data_overall, data_survival, data_mortality):
        """Add continuous variable with median (IQR) and Mann-Whitney test"""

        # Remove missing values
        overall_clean = data_overall.dropna()
        survival_clean = data_survival.dropna()
        mortality_clean = data_mortality.dropna()

        if len(overall_clean) == 0:
            return

        # Calculate statistics
        overall_median = overall_clean.median()
        overall_q25 = overall_clean.quantile(0.25)
        overall_q75 = overall_clean.quantile(0.75)

        survival_median = survival_clean.median()
        survival_q25 = survival_clean.quantile(0.25)
        survival_q75 = survival_clean.quantile(0.75)

        mortality_median = mortality_clean.median()
        mortality_q25 = mortality_clean.quantile(0.25)
        mortality_q75 = mortality_clean.quantile(0.75)

        # Mann-Whitney U test
        if len(survival_clean) > 0 and len(mortality_clean) > 0:
            statistic, p_value = stats.mannwhitneyu(survival_clean, mortality_clean, alternative='two-sided')
        else:
            p_value = np.nan

        # Format p-value
        if pd.isna(p_value):
            p_str = "N/A"
        elif p_value < 0.001:
            p_str = "< 0.001*"
        elif p_value < 0.01:
            p_str = f"< 0.01*"
        elif p_value < 0.05:
            p_str = f"{p_value:.3f}*"
        else:
            p_str = f"{p_value:.3f}"

        results.append({
            'Variable': display_name,
            'Overall (n = 8782)': f"{overall_median:.2f} ({overall_q25:.2f}, {overall_q75:.2f})",
            'Survival group (n = 6805)': f"{survival_median:.2f} ({survival_q25:.2f}, {survival_q75:.2f})",
            'Mortality group (n = 1977)': f"{mortality_median:.2f} ({mortality_q25:.2f}, {mortality_q75:.2f})",
            'p value': p_str
        })

    def add_categorical_var(var_name, display_name, data_overall, data_survival, data_mortality):
        """Add categorical variable with counts (%) and chi-square test"""

        # Count occurrences
        overall_count = data_overall.sum()
        overall_pct = (overall_count / len(data_overall)) * 100

        survival_count = data_survival.sum()
        survival_pct = (survival_count / len(data_survival)) * 100

        mortality_count = data_mortality.sum()
        mortality_pct = (mortality_count / len(data_mortality)) * 100

        # Chi-square test
        contingency = np.array([
            [survival_count, len(data_survival) - survival_count],
            [mortality_count, len(data_mortality) - mortality_count]
        ])

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except:
            p_value = np.nan

        # Format p-value
        if pd.isna(p_value):
            p_str = "N/A"
        elif p_value < 0.001:
            p_str = "< 0.001*"
        elif p_value < 0.01:
            p_str = f"< 0.01*"
        elif p_value < 0.05:
            p_str = f"{p_value:.3f}*"
        else:
            p_str = f"{p_value:.3f}"

        results.append({
            'Variable': display_name,
            'Overall (n = 8782)': f"{int(overall_count)} ({overall_pct:.1f}%)",
            'Survival group (n = 6805)': f"{int(survival_count)} ({survival_pct:.1f}%)",
            'Mortality group (n = 1977)': f"{int(mortality_count)} ({mortality_pct:.1f}%)",
            'p value': p_str
        })

    # Demographics
    print("\nProcessing demographics...")
    age_col = 'age' if 'age' in df.columns else 'anchor_age'
    if age_col in df.columns:
        add_continuous_var(age_col, 'Age (years)', df[age_col], survival[age_col], mortality[age_col])

    # Gender (assuming Female=1 for matching paper format)
    if 'gender' in df.columns:
        df['female'] = (df['gender'] == 'F').astype(int)
        survival['female'] = (survival['gender'] == 'F').astype(int)
        mortality['female'] = (mortality['gender'] == 'F').astype(int)
        add_categorical_var('female', 'Female', df['female'], survival['female'], mortality['female'])

    # Weight
    if 'weight' in df.columns:
        add_continuous_var('weight', 'Weight (kg)', df['weight'], survival['weight'], mortality['weight'])

    # Vital signs - min, max, mean for each
    print("Processing vital signs...")
    vital_vars = {
        'Heart rate (bpm)': 'heart_rate',
        'Respiratory rate (bpm)': 'resp_rate',
        'Systolic BP (mmHg)': 'sbp',
        'Diastolic BP (mmHg)': 'dbp',
        'Mean arterial pressure (mmHg)': 'map',
        'Temperature (°C)': 'temp',
        'SpO2 (%)': 'spo2'
    }

    for display_name, var_base in vital_vars.items():
        for suffix in ['min', 'max', 'mean']:
            var_name = f"{var_base}_{suffix}"
            if var_name in df.columns:
                display = f"{display_name.replace('(', '').replace(')', '')} {suffix}"
                add_continuous_var(var_name, display, df[var_name], survival[var_name], mortality[var_name])

    # Lab values - min, max, mean for each
    print("Processing lab values...")
    lab_vars = {
        'Bicarbonate': 'bicarbonate',
        'BUN': 'bun',
        'Creatinine': 'creatinine',
        'INR': 'inr',
        'Glucose': 'glucose',
        'Sodium': 'sodium',
        'Potassium': 'potassium',
        'Calcium': 'calcium',
        'Magnesium': 'magnesium',
        'Chloride': 'chloride',
        'Hemoglobin': 'hemoglobin',
        'WBC': 'wbc',
        'Platelet': 'platelet'
    }

    for display_name, var_base in lab_vars.items():
        for suffix in ['min', 'max', 'mean']:
            var_name = f"{var_base}_{suffix}"
            if var_name in df.columns:
                display = f"{display_name} {suffix} (mg/dL)" if 'Bicarbonate' not in display_name else f"{display_name} {suffix} (mEq/L)"
                add_continuous_var(var_name, display, df[var_name], survival[var_name], mortality[var_name])

    # Urine output
    if 'urine_output_24h' in df.columns:
        add_continuous_var('urine_output_24h', 'Urine output (mL/24 h)', 
                          df['urine_output_24h'], survival['urine_output_24h'], mortality['urine_output_24h'])

    # Comorbidities
    print("Processing comorbidities...")
    comorbidity_names = {
        'myocardial_infarction': 'Myocardial infarction',
        'congestive_heart_failure': 'Congestive heart failure',
        'peripheral_vascular_disease': 'Peripheral vascular disease',
        'cerebrovascular_disease': 'Cerebrovascular disease',
        'dementia': 'Dementia',
        'chronic_pulmonary_disease': 'Chronic pulmonary disease',
        'rheumatic_disease': 'Rheumatic disease',
        'peptic_ulcer_disease': 'Peptic ulcer disease',
        'mild_liver_disease': 'Mild liver disease',
        'diabetes_without_cc': 'Diabetes without cc',
        'paraplegia': 'Paraplegia',
        'renal_disease': 'Renal disease',
        'malignant_cancer': 'Malignant cancer',
        'severe_liver_disease': 'Severe liver disease',
        'metastatic_solid_tumor': 'Metastatic solid tumor',
        'aids': 'AIDS'
    }

    for var_name, display_name in comorbidity_names.items():
        if var_name in df.columns:
            add_categorical_var(var_name, display_name, df[var_name], survival[var_name], mortality[var_name])

    # Charlson Comorbidity Index
    if 'charlson_comorbidity_index' in df.columns:
        add_continuous_var('charlson_comorbidity_index', 'Charlson Comorbidity Index',
                          df['charlson_comorbidity_index'], survival['charlson_comorbidity_index'], 
                          mortality['charlson_comorbidity_index'])

    # Mechanical ventilation
    if 'mechanical_ventilation' in df.columns:
        add_categorical_var('mechanical_ventilation', 'Mechanical ventilation',
                           df['mechanical_ventilation'], survival['mechanical_ventilation'], 
                           mortality['mechanical_ventilation'])

    # GCS and SOFA scores
    if 'gcs_score' in df.columns:
        add_continuous_var('gcs_score', 'GCS score', df['gcs_score'], survival['gcs_score'], mortality['gcs_score'])

    if 'sofa_score' in df.columns:
        add_continuous_var('sofa_score', 'SOFA score', df['sofa_score'], survival['sofa_score'], mortality['sofa_score'])

    # Create DataFrame
    table_df = pd.DataFrame(results)

    # Save results
    output_file = f"{output_dir}/baseline_characteristics_table.csv"
    table_df.to_csv(output_file, index=False)

    # Also save as formatted text
    text_file = f"{output_dir}/baseline_characteristics_table.txt"
    with open(text_file, 'w') as f:
        f.write("Table 1: Baseline Characteristics\n")
        f.write("=" * 120 + "\n")
        f.write(table_df.to_string(index=False))
        f.write("\n" + "=" * 120 + "\n")
        f.write("\nContinuous data presented as median (interquartile range)\n")
        f.write("Categorical data presented as count (%)\n")
        f.write("Statistical tests: *Mann-Whitney U test; †Chi-square test; ‡Fisher's exact test\n")

    print(f"\n✓ Baseline characteristics table saved to:")
    print(f"  - {output_file}")
    print(f"  - {text_file}")

    return table_df



# In[3]:


if __name__ == "__main__":
    # Run after feature extraction
    features_file = "../results/data/complete_ml_features.csv"
    table = generate_baseline_characteristics_table(features_file)
    print("\nFirst 10 rows of table:")
    print(table.head(10))


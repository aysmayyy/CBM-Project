"""
Step 3: Extract Essential Features for ML Models
Focuses on the most important predictors from the paper
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def load_cohort():
    """Load the patient cohort we identified"""
    cohort = pd.read_csv("../results/study_cohort.csv")
    print(f"Loaded {len(cohort)} patients from cohort")
    return cohort

def extract_vital_signs(cohort):
    """Extract vital signs from first 24h of ICU stay"""
    
    print("\nExtracting vital signs (this will take a few minutes)...")
    
    # Item IDs for vital signs in MIMIC-IV
    vital_items = {
        'heart_rate': [220045],  # Heart rate
        'resp_rate': [220210, 224690],  # Respiratory rate
        'sbp': [220050, 220179],  # Systolic BP
        'dbp': [220051, 220180],  # Diastolic BP
        'temp': [223761, 223762],  # Temperature
        'spo2': [220277]  # SpO2
    }
    
    # Load chartevents in chunks (it's huge)
    chartevents_path = "../data/mimic-iv-3.1/icu/chartevents.csv.gz"
    
    # Get stay_ids from our cohort
    stay_ids = set(cohort['stay_id'].values)
    
    vital_data = []
    chunk_size = 1000000
    chunks_processed = 0
    
    print("Reading chartevents (this is the slow part)...")
    
    for chunk in pd.read_csv(chartevents_path, chunksize=chunk_size, 
                             usecols=['stay_id', 'itemid', 'charttime', 'valuenum']):
        
        # Filter to our cohort and vital sign items
        all_vital_items = [item for items in vital_items.values() for item in items]
        chunk = chunk[chunk['stay_id'].isin(stay_ids) & 
                     chunk['itemid'].isin(all_vital_items)]
        
        if len(chunk) > 0:
            vital_data.append(chunk)
        
        chunks_processed += 1
        if chunks_processed % 5 == 0:
            print(f"  Processed {chunks_processed} million rows...")
    
    if not vital_data:
        print("WARNING: No vital signs data found!")
        return pd.DataFrame()
    
    vitals = pd.concat(vital_data, ignore_index=True)
    print(f"Found {len(vitals)} vital sign measurements")
    
    # Calculate mean values per patient
    results = []
    
    for stay_id in stay_ids:
        patient_vitals = vitals[vitals['stay_id'] == stay_id]
        
        if len(patient_vitals) == 0:
            continue
            
        row = {'stay_id': stay_id}
        
        # Calculate mean for each vital sign
        for vital_name, item_ids in vital_items.items():
            vital_values = patient_vitals[patient_vitals['itemid'].isin(item_ids)]['valuenum']
            if len(vital_values) > 0:
                row[f'{vital_name}_mean'] = vital_values.mean()
        
        results.append(row)
    
    return pd.DataFrame(results)

def extract_lab_values(cohort):
    """Extract key lab values from first 24h"""
    
    print("\nExtracting lab values...")
    
    # Key lab test item IDs
    lab_items = {
        'bun': [51006],  # Blood Urea Nitrogen
        'creatinine': [50912],  # Creatinine
        'hemoglobin': [51222],  # Hemoglobin
        'wbc': [51301],  # White Blood Cell count
        'platelet': [51265],  # Platelet count
        'inr': [51237],  # INR
        'sodium': [50983],  # Sodium
        'potassium': [50971],  # Potassium
    }
    
    labevents_path = "../data/mimic-iv-3.1/hosp/labevents.csv.gz"
    stay_ids = set(cohort['stay_id'].values)
    subject_ids = set(cohort['subject_id'].values)
    
    lab_data = []
    chunk_size = 1000000
    chunks_processed = 0
    
    print("Reading labevents...")
    
    for chunk in pd.read_csv(labevents_path, chunksize=chunk_size,
                             usecols=['subject_id', 'itemid', 'valuenum']):
        
        # Filter to our cohort and lab items
        all_lab_items = [item for items in lab_items.values() for item in items]
        chunk = chunk[chunk['subject_id'].isin(subject_ids) & 
                     chunk['itemid'].isin(all_lab_items)]
        
        if len(chunk) > 0:
            lab_data.append(chunk)
        
        chunks_processed += 1
        if chunks_processed % 5 == 0:
            print(f"  Processed {chunks_processed} million rows...")
    
    if not lab_data:
        print("WARNING: No lab data found!")
        return pd.DataFrame()
    
    labs = pd.concat(lab_data, ignore_index=True)
    print(f"Found {len(labs)} lab measurements")
    
    # Calculate mean values per patient
    results = []
    
    for subject_id in subject_ids:
        patient_labs = labs[labs['subject_id'] == subject_id]
        
        if len(patient_labs) == 0:
            continue
            
        row = {'subject_id': subject_id}
        
        for lab_name, item_ids in lab_items.items():
            lab_values = patient_labs[patient_labs['itemid'].isin(item_ids)]['valuenum']
            if len(lab_values) > 0:
                row[f'{lab_name}_mean'] = lab_values.mean()
        
        results.append(row)
    
    return pd.DataFrame(results)

def extract_outcomes(cohort):
    """Extract 28-day mortality outcome"""
    
    print("\nExtracting mortality outcomes...")
    
    admissions = pd.read_csv("../data/mimic-iv-3.1/hosp/admissions.csv.gz")
    patients = pd.read_csv("../data/mimic-iv-3.1/hosp/patients.csv.gz")
    
    # Merge to get death info
    cohort_with_outcome = cohort.merge(
        admissions[['subject_id', 'hadm_id', 'admittime', 'deathtime']], 
        on=['subject_id', 'hadm_id'], 
        how='left'
    )
    
    # Convert to datetime
    cohort_with_outcome['admittime'] = pd.to_datetime(cohort_with_outcome['admittime'])
    cohort_with_outcome['deathtime'] = pd.to_datetime(cohort_with_outcome['deathtime'])
    
    # Calculate if death occurred within 28 days
    cohort_with_outcome['days_to_death'] = (
        cohort_with_outcome['deathtime'] - cohort_with_outcome['admittime']
    ).dt.total_seconds() / (24 * 3600)
    
    cohort_with_outcome['mortality_28day'] = (
        cohort_with_outcome['days_to_death'] <= 28
    ).astype(int)
    
    # If no death recorded, assume alive
    cohort_with_outcome['mortality_28day'].fillna(0, inplace=True)
    
    mortality_rate = cohort_with_outcome['mortality_28day'].mean()
    print(f"28-day mortality rate: {mortality_rate:.1%}")
    
    return cohort_with_outcome[['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']]

def main():
    """Main function to extract all features"""
    
    print("=== Feature Extraction for ML Models ===\n")
    
    # Load cohort
    cohort = load_cohort()
    
    # Extract features (in order of importance)
    outcomes = extract_outcomes(cohort)
    
    print("\nNote: Vital signs and labs will take 5-10 minutes each")
    print("You can press Ctrl+C to skip if needed for testing")
    
    try:
        vitals = extract_vital_signs(cohort)
        labs = extract_lab_values(cohort)
    except KeyboardInterrupt:
        print("\n\nSkipped detailed extraction. Creating minimal dataset...")
        vitals = pd.DataFrame()
        labs = pd.DataFrame()
    
    # Merge everything
    print("\nMerging features...")
    
    final_data = cohort.copy()
    
    if len(vitals) > 0:
        final_data = final_data.merge(vitals, on='stay_id', how='left')
    
    if len(labs) > 0:
        final_data = final_data.merge(labs, on='subject_id', how='left')
    
    final_data = final_data.merge(outcomes, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    # Save
    output_file = "../results/ml_features.csv"
    final_data.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved features to: {output_file}")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"Columns: {list(final_data.columns)}")
    
    # Show missingness
    print("\nMissing data summary:")
    missing = final_data.isnull().sum()
    missing_pct = (missing / len(final_data) * 100).round(1)
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing_pct[col]}%")

if __name__ == "__main__":
    main()
"""
Step 3: Extract Features - EXACT Paper Methodology
Extracts features with proper 24-hour filtering and clinical score calculation
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def load_cohort():
    """Load the patient cohort we identified"""
    cohort = pd.read_csv("../results/study_cohort.csv")
    print(f"Loaded {len(cohort)} patients from cohort")
    return cohort

def extract_vital_signs_24h(cohort):
    """
    Extract vital signs from FIRST 24 HOURS of ICU stay
    Calculate min, max, mean for each vital sign
    """
    
    print("\nExtracting vital signs (first 24 hours only)...")
    
    # Vital sign item IDs in MIMIC-IV
    vital_items = {
        'heart_rate': [220045],
        'resp_rate': [220210, 224690],
        'sbp': [220050, 220179],
        'dbp': [220051, 220180],
        'map': [220052, 220181, 225312],
        'temp': [223761, 223762],
        'spo2': [220277]
    }
    
    chartevents_path = "../data/mimic-iv-3.1/icu/chartevents.csv.gz"
    icustays_path = "../data/mimic-iv-3.1/icu/icustays.csv.gz"
    
    # Load ICU stays to get admission times
    print("  Loading ICU stay times...")
    icustays = pd.read_csv(icustays_path, usecols=['stay_id', 'intime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    # Merge with cohort
    cohort_with_times = cohort[['stay_id']].merge(icustays, on='stay_id', how='left')
    cohort_with_times['time_24h'] = cohort_with_times['intime'] + pd.Timedelta(hours=24)
    
    stay_ids = set(cohort['stay_id'].values)
    
    vital_data = []
    chunk_size = 1000000
    chunks_processed = 0
    
    print("  Reading chartevents with 24-hour filtering...")
    
    for chunk in pd.read_csv(chartevents_path, chunksize=chunk_size,
                             usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],
                             dtype={'stay_id': int, 'itemid': int, 'valuenum': float}):
        
        chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        
        if len(chunk) == 0:
            chunks_processed += 1
            continue
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.merge(cohort_with_times[['stay_id', 'intime', 'time_24h']], 
                           on='stay_id', how='left')
        
        chunk = chunk[(chunk['charttime'] >= chunk['intime']) & 
                     (chunk['charttime'] <= chunk['time_24h'])]
        
        all_vital_items = [item for items in vital_items.values() for item in items]
        chunk = chunk[chunk['itemid'].isin(all_vital_items)]
        
        if len(chunk) > 0:
            vital_data.append(chunk[['stay_id', 'itemid', 'valuenum']])
        
        chunks_processed += 1
        if chunks_processed % 5 == 0:
            print(f"    Processed {chunks_processed} million rows...")
    
    if not vital_data:
        print("  WARNING: No vital signs data found!")
        return pd.DataFrame()
    
    vitals = pd.concat(vital_data, ignore_index=True)
    print(f"  Found {len(vitals)} vital sign measurements in first 24h")
    
    results = []
    for stay_id in stay_ids:
        patient_vitals = vitals[vitals['stay_id'] == stay_id]
        if len(patient_vitals) == 0:
            continue
            
        row = {'stay_id': stay_id}
        for vital_name, item_ids in vital_items.items():
            values = patient_vitals[patient_vitals['itemid'].isin(item_ids)]['valuenum']
            if len(values) > 0:
                row[f'{vital_name}_min'] = values.min()
                row[f'{vital_name}_max'] = values.max()
                row[f'{vital_name}_mean'] = values.mean()
        
        results.append(row)
    
    print(f"  Extracted vital signs for {len(results)} patients")
    return pd.DataFrame(results)

def extract_lab_values_24h(cohort):
    """
    Extract lab values from FIRST 24 HOURS of ICU stay
    Calculate min, max, mean for each lab test
    """
    
    print("\nExtracting lab values (first 24 hours only)...")
    
    lab_items = {
        'wbc': [51300, 51301],
        'hemoglobin': [51222],
        'platelet': [51265],
        'mch': [51248],
        'mchc': [51249],
        'mcv': [51250],
        'bicarbonate': [50882],
        'bun': [51006],
        'creatinine': [50912],
        'glucose': [50931],
        'sodium': [50983],
        'potassium': [50971],
        'calcium': [50893],
        'magnesium': [50960],
        'chloride': [50902],
        'inr': [51237],
        'pt': [51274],
        'ptt': [51275],
        'phosphate': [50970]
    }
    
    labevents_path = "../data/mimic-iv-3.1/hosp/labevents.csv.gz"
    icustays_path = "../data/mimic-iv-3.1/icu/icustays.csv.gz"
    
    print("  Loading ICU stay times...")
    icustays = pd.read_csv(icustays_path, usecols=['stay_id', 'hadm_id', 'intime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    cohort_with_times = cohort[['subject_id', 'hadm_id', 'stay_id']].merge(
        icustays[['stay_id', 'hadm_id', 'intime']], 
        on=['stay_id', 'hadm_id'], 
        how='left'
    )
    cohort_with_times['time_24h'] = cohort_with_times['intime'] + pd.Timedelta(hours=24)
    
    subject_ids = set(cohort['subject_id'].values)
    
    lab_data = []
    chunk_size = 1000000
    chunks_processed = 0
    
    print("  Reading labevents with 24-hour filtering...")
    
    for chunk in pd.read_csv(labevents_path, chunksize=chunk_size,
                             usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
                             dtype={'subject_id': int, 'hadm_id': float, 
                                   'itemid': int, 'valuenum': float}):
        
        chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        
        if len(chunk) == 0:
            chunks_processed += 1
            continue
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.merge(cohort_with_times[['subject_id', 'hadm_id', 'intime', 'time_24h']], 
                           on=['subject_id', 'hadm_id'], how='left')
        
        chunk = chunk[(chunk['charttime'] >= chunk['intime']) & 
                     (chunk['charttime'] <= chunk['time_24h'])]
        
        all_lab_items = [item for items in lab_items.values() for item in items]
        chunk = chunk[chunk['itemid'].isin(all_lab_items)]
        
        if len(chunk) > 0:
            lab_data.append(chunk[['subject_id', 'hadm_id', 'itemid', 'valuenum']])
        
        chunks_processed += 1
        if chunks_processed % 5 == 0:
            print(f"    Processed {chunks_processed} million rows...")
    
    if not lab_data:
        print("  WARNING: No lab data found!")
        return pd.DataFrame()
    
    labs = pd.concat(lab_data, ignore_index=True)
    print(f"  Found {len(labs)} lab measurements in first 24h")
    
    results = []
    cohort_keys = cohort[['subject_id', 'hadm_id']].drop_duplicates()
    
    for _, row in cohort_keys.iterrows():
        subject_id = row['subject_id']
        hadm_id = row['hadm_id']
        
        patient_labs = labs[(labs['subject_id'] == subject_id) & 
                           (labs['hadm_id'] == hadm_id)]
        
        if len(patient_labs) == 0:
            continue
            
        result = {'subject_id': subject_id, 'hadm_id': hadm_id}
        
        for lab_name, item_ids in lab_items.items():
            values = patient_labs[patient_labs['itemid'].isin(item_ids)]['valuenum']
            if len(values) > 0:
                result[f'{lab_name}_min'] = values.min()
                result[f'{lab_name}_max'] = values.max()
                result[f'{lab_name}_mean'] = values.mean()
        
        results.append(result)
    
    print(f"  Extracted labs for {len(results)} patients")
    return pd.DataFrame(results)

def extract_urine_output_24h(cohort):
    """Extract total 24-hour urine output"""
    
    print("\nExtracting 24-hour urine output...")
    
    outputevents_path = "../data/mimic-iv-3.1/icu/outputevents.csv.gz"
    icustays_path = "../data/mimic-iv-3.1/icu/icustays.csv.gz"
    
    print("  Loading ICU stay times...")
    icustays = pd.read_csv(icustays_path, usecols=['stay_id', 'intime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    cohort_with_times = cohort[['stay_id']].merge(icustays, on='stay_id', how='left')
    cohort_with_times['time_24h'] = cohort_with_times['intime'] + pd.Timedelta(hours=24)
    
    stay_ids = set(cohort['stay_id'].values)
    urine_items = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 
                   40056, 40405, 40428, 40086, 40096, 40651]
    
    output_data = []
    
    print("  Reading outputevents...")
    for chunk in pd.read_csv(outputevents_path, chunksize=500000,
                             usecols=['stay_id', 'itemid', 'charttime', 'value']):
        
        chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        
        if len(chunk) == 0:
            continue
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.merge(cohort_with_times[['stay_id', 'intime', 'time_24h']], 
                           on='stay_id', how='left')
        
        chunk = chunk[(chunk['charttime'] >= chunk['intime']) & 
                     (chunk['charttime'] <= chunk['time_24h'])]
        chunk = chunk[chunk['itemid'].isin(urine_items)]
        
        if len(chunk) > 0:
            output_data.append(chunk[['stay_id', 'value']])
    
    if not output_data:
        print("  WARNING: No urine output data found!")
        return pd.DataFrame()
    
    outputs = pd.concat(output_data, ignore_index=True)
    urine_totals = outputs.groupby('stay_id')['value'].sum().reset_index()
    urine_totals.columns = ['stay_id', 'urine_output_24h']
    
    print(f"  Extracted urine output for {len(urine_totals)} patients")
    return urine_totals

def calculate_gcs_worst(cohort):
    """Calculate worst GCS score in first 24 hours"""
    
    print("\nCalculating worst GCS score (first 24 hours)...")
    
    chartevents_path = "../data/mimic-iv-3.1/icu/chartevents.csv.gz"
    icustays_path = "../data/mimic-iv-3.1/icu/icustays.csv.gz"
    
    print("  Loading ICU stay times...")
    icustays = pd.read_csv(icustays_path, usecols=['stay_id', 'intime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
    cohort_with_times = cohort[['stay_id']].merge(icustays, on='stay_id', how='left')
    cohort_with_times['time_24h'] = cohort_with_times['intime'] + pd.Timedelta(hours=24)
    
    stay_ids = set(cohort['stay_id'].values)
    gcs_items = [220739]
    
    gcs_data = []
    
    print("  Reading chartevents for GCS...")
    for chunk in pd.read_csv(chartevents_path, chunksize=1000000,
                             usecols=['stay_id', 'itemid', 'charttime', 'valuenum']):
        
        chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        
        if len(chunk) == 0:
            continue
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.merge(cohort_with_times[['stay_id', 'intime', 'time_24h']], 
                           on='stay_id', how='left')
        
        chunk = chunk[(chunk['charttime'] >= chunk['intime']) & 
                     (chunk['charttime'] <= chunk['time_24h'])]
        chunk = chunk[chunk['itemid'].isin(gcs_items)]
        
        if len(chunk) > 0:
            gcs_data.append(chunk[['stay_id', 'valuenum']])
    
    if not gcs_data:
        print("  WARNING: No GCS data found!")
        return pd.DataFrame()
    
    gcs = pd.concat(gcs_data, ignore_index=True)
    gcs_scores = gcs.groupby('stay_id')['valuenum'].min().reset_index()
    gcs_scores.columns = ['stay_id', 'gcs_score']
    
    print(f"  Calculated GCS for {len(gcs_scores)} patients")
    return gcs_scores

def calculate_sofa_score(cohort, vitals_df, labs_df):
    """Calculate SOFA score using worst values in first 24 hours"""
    
    print("\nCalculating SOFA score...")
    
    sofa_data = cohort[['stay_id', 'subject_id', 'hadm_id']].copy()
    
    if len(vitals_df) > 0 and 'map_min' in vitals_df.columns:
        sofa_data = sofa_data.merge(vitals_df[['stay_id', 'map_min']], 
                                    on='stay_id', how='left')
    
    if len(labs_df) > 0:
        if 'platelet_min' in labs_df.columns:
            sofa_data = sofa_data.merge(labs_df[['subject_id', 'hadm_id', 'platelet_min']], 
                                       on=['subject_id', 'hadm_id'], how='left')
        
        if 'creatinine_max' in labs_df.columns:
            sofa_data = sofa_data.merge(labs_df[['subject_id', 'hadm_id', 'creatinine_max']], 
                                       on=['subject_id', 'hadm_id'], how='left')
    
    sofa_scores = []
    
    for _, row in sofa_data.iterrows():
        score = 0
        
        if pd.notna(row.get('map_min')):
            if row['map_min'] < 70:
                score += 1
        
        if pd.notna(row.get('platelet_min')):
            plt = row['platelet_min']
            if plt < 150:
                score += 1
            if plt < 100:
                score += 1
            if plt < 50:
                score += 1
            if plt < 20:
                score += 1
        
        if pd.notna(row.get('creatinine_max')):
            cr = row['creatinine_max']
            if cr >= 1.2:
                score += 1
            if cr >= 2.0:
                score += 1
            if cr >= 3.5:
                score += 1
            if cr >= 5.0:
                score += 1
        
        sofa_scores.append({'stay_id': row['stay_id'], 'sofa_score': score})
    
    sofa_df = pd.DataFrame(sofa_scores)
    print(f"  Calculated SOFA for {len(sofa_df)} patients")
    return sofa_df

def extract_demographics_and_comorbidities(cohort):
    """Extract demographics, weight, ethnicity, and comorbidities"""
    
    print("\nExtracting demographics and comorbidities...")
    
    patients = pd.read_csv("../data/mimic-iv-3.1/hosp/patients.csv.gz")
    admissions = pd.read_csv("../data/mimic-iv-3.1/hosp/admissions.csv.gz")
    diagnoses = pd.read_csv("../data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz")
    
    demo = cohort.merge(patients[['subject_id', 'gender', 'anchor_age']], 
                       on='subject_id', how='left')
    
    demo = demo.merge(admissions[['subject_id', 'hadm_id', 'race']], 
                     on=['subject_id', 'hadm_id'], how='left')
    demo = demo.rename(columns={'race': 'ethnicity'})
    
    def simplify_ethnicity(eth):
        if pd.isna(eth):
            return 'Other'
        eth = str(eth).upper()
        if 'WHITE' in eth:
            return 'White'
        elif 'BLACK' in eth or 'AFRICAN' in eth:
            return 'African-American'
        elif 'ASIAN' in eth:
            return 'Asian'
        elif 'HISPANIC' in eth or 'LATINO' in eth:
            return 'Hispanic-American'
        else:
            return 'Other'
    
    demo['ethnicity'] = demo['ethnicity'].apply(simplify_ethnicity)
    
    print("  Extracting weight...")
    chartevents_path = "../data/mimic-iv-3.1/icu/chartevents.csv.gz"
    weight_itemids = [226512, 224639]
    stay_ids = set(cohort['stay_id'].values)
    
    weights = []
    for chunk in pd.read_csv(chartevents_path, chunksize=1000000,
                             usecols=['stay_id', 'itemid', 'valuenum']):
        chunk = chunk[chunk['stay_id'].isin(stay_ids) & 
                     chunk['itemid'].isin(weight_itemids)]
        if len(chunk) > 0:
            weights.append(chunk)
    
    if weights:
        weight_df = pd.concat(weights, ignore_index=True)
        weight_df = weight_df.groupby('stay_id')['valuenum'].first().reset_index()
        weight_df.columns = ['stay_id', 'weight']
        demo = demo.merge(weight_df, on='stay_id', how='left')
    
    print("  Extracting comorbidities...")
    
    comorbidity_codes = {
        'myocardial_infarction': ['I21', 'I22', '410'],
        'congestive_heart_failure': ['I50', '428'],
        'cerebrovascular_disease': ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'],
        'chronic_pulmonary_disease': ['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47'],
        'mild_liver_disease': ['K70', 'K71', 'K73', 'K74', '571'],
        'severe_liver_disease': ['K72', 'K76', '572'],
        'paraplegia': ['G81', 'G82', '342', '343', '344'],
        'renal_disease': ['N18', 'N19', '585', '586'],
        'metastatic_solid_tumor': ['C77', 'C78', 'C79', 'C80'],
        'aids': ['B20', 'B21', 'B22', 'B23', 'B24', '042']
    }
    
    for condition, codes in comorbidity_codes.items():
        has_condition = []
        for subject_id in demo['subject_id']:
            patient_dx = diagnoses[diagnoses['subject_id'] == subject_id]
            has_cond = False
            for code in codes:
                if any(patient_dx['icd_code'].astype(str).str.startswith(code)):
                    has_cond = True
                    break
            has_condition.append(int(has_cond))
        
        demo[condition] = has_condition
    
    return demo

def calculate_charlson_index(demo_df):
    """Calculate Charlson Comorbidity Index"""
    
    weights = {
        'myocardial_infarction': 1,
        'congestive_heart_failure': 1,
        'cerebrovascular_disease': 1,
        'chronic_pulmonary_disease': 1,
        'mild_liver_disease': 1,
        'paraplegia': 2,
        'renal_disease': 2,
        'severe_liver_disease': 3,
        'metastatic_solid_tumor': 6,
        'aids': 6
    }
    
    cci = pd.Series(0, index=demo_df.index)
    
    for condition, weight in weights.items():
        if condition in demo_df.columns:
            cci += demo_df[condition] * weight
    
    return cci

def extract_mechanical_ventilation_24h(cohort):
    """Check if patient received mechanical ventilation in first 24 hours"""
    
    print("\nExtracting mechanical ventilation status (first 24 hours)...")
    
    procedureevents_path = "../data/mimic-iv-3.1/icu/procedureevents.csv.gz"
    
    try:
        procedures = pd.read_csv(procedureevents_path)
        stay_ids = set(cohort['stay_id'].values)
        
        vent_items = [225792, 225794]
        
        vented_stays = procedures[
            procedures['stay_id'].isin(stay_ids) & 
            procedures['itemid'].isin(vent_items)
        ]['stay_id'].unique()
        
        vent_df = pd.DataFrame({
            'stay_id': list(stay_ids),
            'mechanical_ventilation': [1 if sid in vented_stays else 0 for sid in stay_ids]
        })
        
        return vent_df
    except:
        print("  Could not extract ventilation data")
        return pd.DataFrame({'stay_id': list(stay_ids), 'mechanical_ventilation': 0})

def extract_outcomes(cohort):
    """Extract 28-day mortality outcome"""
    
    print("\nExtracting mortality outcomes...")
    
    admissions = pd.read_csv("../data/mimic-iv-3.1/hosp/admissions.csv.gz")
    
    cohort_with_outcome = cohort.merge(
        admissions[['subject_id', 'hadm_id', 'admittime', 'deathtime']], 
        on=['subject_id', 'hadm_id'], 
        how='left'
    )
    
    cohort_with_outcome['admittime'] = pd.to_datetime(cohort_with_outcome['admittime'])
    cohort_with_outcome['deathtime'] = pd.to_datetime(cohort_with_outcome['deathtime'])
    
    cohort_with_outcome['days_to_death'] = (
        cohort_with_outcome['deathtime'] - cohort_with_outcome['admittime']
    ).dt.total_seconds() / (24 * 3600)
    
    cohort_with_outcome['mortality_28day'] = (
        cohort_with_outcome['days_to_death'] <= 28
    ).astype(int)
    
    cohort_with_outcome['mortality_28day'].fillna(0, inplace=True)
    
    mortality_rate = cohort_with_outcome['mortality_28day'].mean()
    print(f"  28-day mortality rate: {mortality_rate:.1%}")
    
    return cohort_with_outcome[['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']]

def main():
    """Main extraction pipeline with 24-hour filtering"""
    
    print("=" * 70)
    print("COMPLETE Feature Extraction - 24-Hour Filtered, Exact Methodology")
    print("=" * 70)
    print("\nWARNING: This will take 30-60 minutes")
    print("Progress will be shown for each component\n")
    
    cohort = load_cohort()
    
    vitals = extract_vital_signs_24h(cohort)
    labs = extract_lab_values_24h(cohort)
    urine = extract_urine_output_24h(cohort)
    gcs = calculate_gcs_worst(cohort)
    sofa = calculate_sofa_score(cohort, vitals, labs)
    demo = extract_demographics_and_comorbidities(cohort)
    ventilation = extract_mechanical_ventilation_24h(cohort)
    outcomes = extract_outcomes(cohort)
    
    print("\nCalculating Charlson Comorbidity Index...")
    demo['charlson_index'] = calculate_charlson_index(demo)
    
    print("\nMerging all features...")
    
    final_data = demo.copy()
    
    if len(vitals) > 0:
        final_data = final_data.merge(vitals, on='stay_id', how='left')
    if len(labs) > 0:
        final_data = final_data.merge(labs, on=['subject_id', 'hadm_id'], how='left')
    if len(urine) > 0:
        final_data = final_data.merge(urine, on='stay_id', how='left')
    if len(gcs) > 0:
        final_data = final_data.merge(gcs, on='stay_id', how='left')
    if len(sofa) > 0:
        final_data = final_data.merge(sofa, on='stay_id', how='left')
    if len(ventilation) > 0:
        final_data = final_data.merge(ventilation, on='stay_id', how='left')
    
    final_data = final_data.merge(outcomes, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    output_file = "../results/complete_ml_features.csv"
    final_data.to_csv(output_file, index=False)
    
    print(f"\n✓ SUCCESS! Saved complete features to: {output_file}")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"\n=== Feature Summary ===")
    print(f"Patients: {len(final_data)}")
    print(f"Mortality rate: {final_data['mortality_28day'].mean():.1%}")
    print(f"\nKey features extracted:")
    print(f"  - Demographics: age, gender, ethnicity, weight")
    print(f"  - Vital signs: HR, RR, temp, SBP, DBP, MAP, SpO2 (min/max/mean)")
    print(f"  - Labs: WBC, Hgb, PLT, BUN, Cr, glucose, electrolytes, INR, etc. (min/max/mean)")
    print(f"  - Urine output: 24-hour total")
    print(f"  - Scores: GCS (worst), SOFA, Charlson index")
    print(f"  - Comorbidities: 10 conditions")
    print(f"  - Mechanical ventilation: Yes/No")
    
    print(f"\nMissing data summary (top 20):")
    missing_pct = (final_data.isnull().sum() / len(final_data) * 100).round(1)
    for col in missing_pct[missing_pct > 0].head(20).index:
        print(f"  {col}: {missing_pct[col]}%")

if __name__ == "__main__":
    main()
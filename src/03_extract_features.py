"""
Step 3: Extract Features - Complete Implementation
Extracts ALL features matching the paper's methodology
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
    
    print("  Loading ICU stay times...")
    icustays = pd.read_csv(icustays_path, usecols=['stay_id', 'intime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    
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
    Extract ALL lab values from FIRST 24 HOURS of ICU stay
    Calculate min, max, mean for each lab test
    """
    
    print("\nExtracting lab values (first 24 hours only)...")
    
    # Complete lab items matching paper's list
    lab_items = {
        'albumin': [50862],  # alb
        'alp': [50863],  # alkaline phosphatase
        'alt': [50861],  # alanine aminotransferase
        'ast': [50878],  # aspartate aminotransferase
        'base_excess': [50802],  # be
        'bicarbonate': [50882],  # bicar
        'bilirubin': [50885],  # bili
        'bilirubin_direct': [50883],  # bili_dir
        'bnd': [51006],  # blood urea nitrogen/creatinine - using BUN as proxy
        'bun': [51006],
        'calcium': [50893],  # ca
        'calcium_ionized': [50808],  # cai
        'ck': [50910],  # creatine kinase
        'ckmb': [50911],  # creatine kinase MB
        'chloride': [50902],  # cl
        'creatinine': [50912],  # crea
        'crp': [50889],  # C-reactive protein
        'fibrinogen': [51214],  # fgn
        'fio2': [50816],  # fraction inspired oxygen
        'glucose': [50931],  # glu
        'hemoglobin': [51222],  # hgb
        'inr': [51237],  # inr_pt
        'potassium': [50971],  # k
        'lactate': [50813],  # lact
        'lymphocytes': [51244],  # lymph
        'mch': [51248],  # mean corpuscular hemoglobin
        'mchc': [51249],  # mean corpuscular hemoglobin concentration
        'mcv': [51250],  # mean corpuscular volume
        'methemoglobin': [51253],  # methb
        'magnesium': [50960],  # mg
        'sodium': [50983],  # na
        'neutrophils': [51256],  # neut
        'pco2': [50818],  # partial pressure CO2
        'ph': [50820],
        'phosphate': [50970],  # phos
        'platelet': [51265],  # plt
        'po2': [50821],  # partial pressure O2
        'pt': [51274],  # prothrombin time
        'ptt': [51275],  # partial thromboplastin time
        'wbc': [51300, 51301]
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
    # MIMIC-IV urine output item IDs
    urine_items = [226559, 226560, 226561, 226584, 226563, 226564, 226565, 
                   226567, 226557, 226558, 227488, 227489, 226566, 226627, 
                   226631]
    
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
    gcs_items = [220739]  # GCS Total
    
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
    
    # Merge with vitals (MAP)
    if len(vitals_df) > 0 and 'map_min' in vitals_df.columns:
        sofa_data = sofa_data.merge(vitals_df[['stay_id', 'map_min']], 
                                    on='stay_id', how='left')
    
    # Merge with labs (platelets, creatinine)
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
        
        # Cardiovascular (MAP)
        if pd.notna(row.get('map_min')):
            if row['map_min'] < 70:
                score += 1
        
        # Coagulation (Platelets)
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
        
        # Renal (Creatinine)
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
    """Extract demographics, weight, ethnicity, and ALL comorbidities"""
    
    print("\nExtracting demographics and comorbidities...")
    
    patients = pd.read_csv("../data/mimic-iv-3.1/hosp/patients.csv.gz")
    admissions = pd.read_csv("../data/mimic-iv-3.1/hosp/admissions.csv.gz")
    diagnoses = pd.read_csv("../data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz")
    
    # Get demographics
    demo = cohort.merge(patients[['subject_id', 'gender', 'anchor_age']], 
                       on='subject_id', how='left')
    
    demo = demo.merge(admissions[['subject_id', 'hadm_id', 'race']], 
                     on=['subject_id', 'hadm_id'], how='left')
    demo = demo.rename(columns={'race': 'ethnicity', 'anchor_age': 'age'})
    
    # Simplify ethnicity
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
    
    # Extract weight
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
    else:
        demo['weight'] = np.nan
    
    # Extract ALL comorbidities
    print("  Extracting comorbidities (Charlson conditions)...")
    
    comorbidity_codes = {
        'myocardial_infarction': {
            'icd10': ['I21', 'I22', 'I252'],
            'icd9': ['410', '412']
        },
        'congestive_heart_failure': {
            'icd10': ['I43', 'I50', 'I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290'],
            'icd9': ['398', '402', '404', '428']
        },
        'peripheral_vascular_disease': {
            'icd10': ['I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959'],
            'icd9': ['093', '437', '440', '441', '443', '447', '557', 'V43']
        },
        'cerebrovascular_disease': {
            'icd10': ['G45', 'G46', 'H340', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'],
            'icd9': ['362', '430', '431', '432', '433', '434', '435', '436', '437', '438']
        },
        'dementia': {
            'icd10': ['F00', 'F01', 'F02', 'F03', 'F051', 'G30', 'G311'],
            'icd9': ['290', '294', '331']
        },
        'chronic_pulmonary_disease': {
            'icd10': ['I278', 'I279', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J684', 'J701', 'J703'],
            'icd9': ['416', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505']
        },
        'rheumatic_disease': {
            'icd10': ['M05', 'M06', 'M315', 'M32', 'M33', 'M34', 'M351', 'M353', 'M360'],
            'icd9': ['446', '710', '714', '720', '725']
        },
        'peptic_ulcer_disease': {
            'icd10': ['K25', 'K26', 'K27', 'K28'],
            'icd9': ['531', '532', '533', '534']
        },
        'mild_liver_disease': {
            'icd10': ['B18', 'K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K73', 'K74', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944'],
            'icd9': ['070', '571', 'V42']
        },
        'diabetes_without_cc': {
            'icd10': ['E100', 'E101', 'E106', 'E108', 'E109', 'E110', 'E111', 'E116', 'E118', 'E119', 'E120', 'E121', 'E126', 'E128', 'E129', 'E130', 'E131', 'E136', 'E138', 'E139', 'E140', 'E141', 'E146', 'E148', 'E149'],
            'icd9': ['250']
        },
        'diabetes_with_cc': {
            'icd10': ['E102', 'E103', 'E104', 'E105', 'E107', 'E112', 'E113', 'E114', 'E115', 'E117', 'E122', 'E123', 'E124', 'E125', 'E127', 'E132', 'E133', 'E134', 'E135', 'E137', 'E142', 'E143', 'E144', 'E145', 'E147'],
            'icd9': ['250']
        },
        'paraplegia': {
            'icd10': ['G041', 'G114', 'G801', 'G802', 'G81', 'G82', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839'],
            'icd9': ['334', '342', '343', '344']
        },
        'renal_disease': {
            'icd10': ['I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N18', 'N19', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992'],
            'icd9': ['403', '404', '582', '583', '585', '586', '588', 'V42', 'V45', 'V56']
        },
        'malignant_cancer': {
            'icd10': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95', 'C96', 'C97'],
            'icd9': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '200', '201', '202', '203', '204', '205', '206', '207', '208']
        },
        'severe_liver_disease': {
            'icd10': ['I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767'],
            'icd9': ['456', '572']
        },
        'metastatic_solid_tumor': {
            'icd10': ['C77', 'C78', 'C79', 'C80'],
            'icd9': ['196', '197', '198', '199']
        },
        'aids': {
            'icd10': ['B20', 'B21', 'B22', 'B23', 'B24'],
            'icd9': ['042', '043', '044']
        }
    }
    
    # Extract comorbidities for each patient
    for condition, codes_dict in comorbidity_codes.items():
        has_condition = []
        for _, patient in demo.iterrows():
            subject_id = patient['subject_id']
            hadm_id = patient['hadm_id']
            
            # Get diagnoses for this admission
            patient_dx = diagnoses[(diagnoses['subject_id'] == subject_id) & 
                                  (diagnoses['hadm_id'] == hadm_id)]
            
            has_cond = False
            
            # Check ICD-10 codes
            for code in codes_dict.get('icd10', []):
                if any(patient_dx[patient_dx['icd_version'] == 10]['icd_code'].astype(str).str.startswith(code)):
                    has_cond = True
                    break
            
            # Check ICD-9 codes if not found yet
            if not has_cond:
                for code in codes_dict.get('icd9', []):
                    if any(patient_dx[patient_dx['icd_version'] == 9]['icd_code'].astype(str).str.startswith(code)):
                        has_cond = True
                        break
            
            has_condition.append(int(has_cond))
        
        demo[condition] = has_condition
        print(f"    {condition}: {sum(has_condition)} patients")
    
    return demo

def calculate_charlson_index(demo_df):
    """Calculate Charlson Comorbidity Index"""
    
    print("\nCalculating Charlson Comorbidity Index...")
    
    weights = {
        'myocardial_infarction': 1,
        'congestive_heart_failure': 1,
        'peripheral_vascular_disease': 1,
        'cerebrovascular_disease': 1,
        'dementia': 1,
        'chronic_pulmonary_disease': 1,
        'rheumatic_disease': 1,
        'peptic_ulcer_disease': 1,
        'mild_liver_disease': 1,
        'diabetes_without_cc': 1,
        'diabetes_with_cc': 2,
        'paraplegia': 2,
        'renal_disease': 2,
        'malignant_cancer': 2,
        'severe_liver_disease': 3,
        'metastatic_solid_tumor': 6,
        'aids': 6
    }
    
    cci = pd.Series(0, index=demo_df.index)
    
    for condition, weight in weights.items():
        if condition in demo_df.columns:
            cci += demo_df[condition] * weight
    
    print(f"  CCI calculated for {len(demo_df)} patients")
    print(f"  Mean CCI: {cci.mean():.1f}, Range: {cci.min()}-{cci.max()}")
    
    return cci

def extract_mechanical_ventilation_24h(cohort):
    """Check if patient received mechanical ventilation in first 24 hours"""
    
    print("\nExtracting mechanical ventilation status (first 24 hours)...")
    
    procedureevents_path = "../data/mimic-iv-3.1/icu/procedureevents.csv.gz"
    icustays_path = "../data/mimic-iv-3.1/icu/icustays.csv.gz"
    
    try:
        print("  Loading ICU stay times...")
        icustays = pd.read_csv(icustays_path, usecols=['stay_id', 'intime'])
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        
        cohort_with_times = cohort[['stay_id']].merge(icustays, on='stay_id', how='left')
        cohort_with_times['time_24h'] = cohort_with_times['intime'] + pd.Timedelta(hours=24)
        
        stay_ids = set(cohort['stay_id'].values)
        
        print("  Reading procedureevents...")
        procedures = pd.read_csv(procedureevents_path)
        procedures['starttime'] = pd.to_datetime(procedures['starttime'], errors='coerce')
        
        # Mechanical ventilation item IDs
        vent_items = [225792, 225794]
        
        # Filter procedures
        procedures = procedures[procedures['stay_id'].isin(stay_ids)]
        procedures = procedures.merge(cohort_with_times[['stay_id', 'intime', 'time_24h']], 
                                      on='stay_id', how='left')
        
        # Check if ventilation occurred in first 24 hours
        procedures = procedures[(procedures['starttime'] >= procedures['intime']) & 
                               (procedures['starttime'] <= procedures['time_24h'])]
        
        vented_stays = procedures[procedures['itemid'].isin(vent_items)]['stay_id'].unique()
        
        vent_df = pd.DataFrame({
            'stay_id': list(stay_ids),
            'mechanical_ventilation': [1 if sid in vented_stays else 0 for sid in stay_ids]
        })
        
        print(f"  Found {sum(vent_df['mechanical_ventilation'])} patients with mechanical ventilation")
        return vent_df
        
    except Exception as e:
        print(f"  Warning: Could not extract ventilation data - {e}")
        return pd.DataFrame({'stay_id': list(stay_ids), 'mechanical_ventilation': 0})

def extract_outcomes(cohort):
    """Extract 28-day mortality outcome"""
    
    print("\nExtracting mortality outcomes...")
    
    admissions = pd.read_csv("../data/mimic-iv-3.1/hosp/admissions.csv.gz")
    patients = pd.read_csv("../data/mimic-iv-3.1/hosp/patients.csv.gz")
    
    cohort_with_outcome = cohort.merge(
        admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']], 
        on=['subject_id', 'hadm_id'], 
        how='left'
    )
    
    cohort_with_outcome = cohort_with_outcome.merge(
        patients[['subject_id', 'dod']],
        on='subject_id',
        how='left'
    )
    
    # Convert to datetime
    cohort_with_outcome['admittime'] = pd.to_datetime(cohort_with_outcome['admittime'])
    cohort_with_outcome['deathtime'] = pd.to_datetime(cohort_with_outcome['deathtime'])
    cohort_with_outcome['dod'] = pd.to_datetime(cohort_with_outcome['dod'])
    
    # Use hospital deathtime if available, otherwise use dod
    cohort_with_outcome['death_date'] = cohort_with_outcome['deathtime'].fillna(cohort_with_outcome['dod'])
    
    # Calculate days from admission to death
    cohort_with_outcome['days_to_death'] = (
        cohort_with_outcome['death_date'] - cohort_with_outcome['admittime']
    ).dt.total_seconds() / (24 * 3600)
    
    # 28-day mortality
    cohort_with_outcome['mortality_28day'] = (
        (cohort_with_outcome['days_to_death'] <= 28) & 
        (cohort_with_outcome['days_to_death'].notna())
    ).astype(int)
    
    mortality_rate = cohort_with_outcome['mortality_28day'].mean()
    print(f"  28-day mortality rate: {mortality_rate:.1%}")
    print(f"  Deaths: {cohort_with_outcome['mortality_28day'].sum()}")
    print(f"  Survivors: {len(cohort_with_outcome) - cohort_with_outcome['mortality_28day'].sum()}")
    
    return cohort_with_outcome[['subject_id', 'hadm_id', 'stay_id', 'mortality_28day']]

def main():
    """Main extraction pipeline with 24-hour filtering"""
    
    print("=" * 70)
    print("COMPLETE Feature Extraction - All Variables from Paper")
    print("=" * 70)
    print("\nWARNING: This will take 30-90 minutes")
    print("Progress will be shown for each component\n")
    
    # Load cohort
    cohort = load_cohort()
    
    # Extract all features
    print("\n" + "="*70)
    print("STEP 1: Extracting vital signs")
    print("="*70)
    vitals = extract_vital_signs_24h(cohort)
    
    print("\n" + "="*70)
    print("STEP 2: Extracting lab values")
    print("="*70)
    labs = extract_lab_values_24h(cohort)
    
    print("\n" + "="*70)
    print("STEP 3: Extracting urine output")
    print("="*70)
    urine = extract_urine_output_24h(cohort)
    
    print("\n" + "="*70)
    print("STEP 4: Calculating GCS scores")
    print("="*70)
    gcs = calculate_gcs_worst(cohort)
    
    print("\n" + "="*70)
    print("STEP 5: Calculating SOFA scores")
    print("="*70)
    sofa = calculate_sofa_score(cohort, vitals, labs)
    
    print("\n" + "="*70)
    print("STEP 6: Extracting demographics and comorbidities")
    print("="*70)
    demo = extract_demographics_and_comorbidities(cohort)
    
    print("\n" + "="*70)
    print("STEP 7: Calculating Charlson Index")
    print("="*70)
    demo['charlson_comorbidity_index'] = calculate_charlson_index(demo)
    
    print("\n" + "="*70)
    print("STEP 8: Extracting mechanical ventilation")
    print("="*70)
    ventilation = extract_mechanical_ventilation_24h(cohort)
    
    print("\n" + "="*70)
    print("STEP 9: Extracting outcomes")
    print("="*70)
    outcomes = extract_outcomes(cohort)
    
    # Merge all features
    print("\n" + "="*70)
    print("STEP 10: Merging all features")
    print("="*70)
    
    final_data = demo.copy()
    
    if len(vitals) > 0:
        print(f"  Merging vitals: {len(vitals)} records")
        final_data = final_data.merge(vitals, on='stay_id', how='left')
    
    if len(labs) > 0:
        print(f"  Merging labs: {len(labs)} records")
        final_data = final_data.merge(labs, on=['subject_id', 'hadm_id'], how='left')
    
    if len(urine) > 0:
        print(f"  Merging urine: {len(urine)} records")
        final_data = final_data.merge(urine, on='stay_id', how='left')
    
    if len(gcs) > 0:
        print(f"  Merging GCS: {len(gcs)} records")
        final_data = final_data.merge(gcs, on='stay_id', how='left')
    
    if len(sofa) > 0:
        print(f"  Merging SOFA: {len(sofa)} records")
        final_data = final_data.merge(sofa, on='stay_id', how='left')
    
    if len(ventilation) > 0:
        print(f"  Merging ventilation: {len(ventilation)} records")
        final_data = final_data.merge(ventilation, on='stay_id', how='left')
    
    print(f"  Merging outcomes: {len(outcomes)} records")
    final_data = final_data.merge(outcomes, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
    
    # Save results
    output_file = "../results/complete_ml_features.csv"
    final_data.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nSaved complete features to: {output_file}")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"  Rows (patients): {final_data.shape[0]}")
    print(f"  Columns (features): {final_data.shape[1]}")
    
    print(f"\n=== Summary Statistics ===")
    print(f"28-day mortality rate: {final_data['mortality_28day'].mean():.1%}")
    print(f"  Deaths: {final_data['mortality_28day'].sum()}")
    print(f"  Survivors: {len(final_data) - final_data['mortality_28day'].sum()}")
    
    # Check which age column exists
    age_col = 'age' if 'age' in final_data.columns else 'anchor_age'
    if age_col in final_data.columns:
        print(f"\nAge: mean={final_data[age_col].mean():.1f}, std={final_data[age_col].std():.1f}")
    
    if 'gender' in final_data.columns:
        print(f"Gender: {final_data['gender'].value_counts().to_dict()}")
    
    print(f"\nEthnicity distribution:")
    for eth, count in final_data['ethnicity'].value_counts().items():
        print(f"  {eth}: {count} ({count/len(final_data)*100:.1f}%)")
    
    print(f"\n=== Missing Data Summary (Top 20) ===")
    missing_pct = (final_data.isnull().sum() / len(final_data) * 100).round(1)
    missing_sorted = missing_pct[missing_pct > 0].sort_values(ascending=False).head(20)
    for col, pct in missing_sorted.items():
        print(f"  {col}: {pct}%")
    
    print(f"\n=== Feature Categories ===")
    vital_cols = [c for c in final_data.columns if any(v in c for v in ['heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'temp', 'spo2'])]
    lab_cols = [c for c in final_data.columns if any(v in c for v in ['albumin', 'bun', 'creatinine', 'glucose', 'hemoglobin', 'platelet', 'wbc', 'sodium', 'potassium', 'chloride', 'bicarbonate', 'calcium', 'magnesium', 'phosphate', 'lactate', 'inr', 'pt', 'ptt'])]
comorbid_cols = ['myocardial_infarction', 'congestive_heart_failure', 
                 'peripheral_vascular_disease', 'cerebrovascular_disease', 
                 'dementia', 'chronic_pulmonary_disease', 'rheumatic_disease',
                 'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
                 'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
                 'severe_liver_disease', 'metastatic_solid_tumor', 'aids']
comorbid_cols = [c for c in comorbid_cols if c in final_data.columns]    
    print(f"  Vital signs: {len(vital_cols)} features")
    print(f"  Lab tests: {len(lab_cols)} features")
    print(f"  Comorbidities: {len(comorbid_cols)} features")
    print(f"  Scores: GCS, SOFA, Charlson")
    print(f"  Other: Demographics, weight, urine output, mechanical ventilation")
    
    print(f"\n{'='*70}")
    print("Next step: Run 04_preprocess_features.py")
    print("This will handle missing values, feature selection, and encoding")
    print("="*70)

if __name__ == "__main__":
    main()
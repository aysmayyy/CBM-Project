"""
Step 2: Identify Immunocompromised Patients
This script finds patients who meet our study criteria
"""
import pandas as pd
import numpy as np

def load_basic_tables():
    """Load the essential MIMIC tables we need"""
    
    data_path = "../data/mimic-iv-3.1"
    
    print("Loading MIMIC-IV tables...")
    
    # Load patients (demographics)
    patients = pd.read_csv(f"{data_path}/hosp/patients.csv.gz")
    print(f"Patients: {len(patients)} records")
    
    # Load admissions 
    admissions = pd.read_csv(f"{data_path}/hosp/admissions.csv.gz")
    print(f"Admissions: {len(admissions)} records")
    
    # Load ICU stays
    icustays = pd.read_csv(f"{data_path}/icu/icustays.csv.gz")
    print(f"ICU stays: {len(icustays)} records")
    
    # Load diagnoses
    diagnoses = pd.read_csv(f"{data_path}/hosp/diagnoses_icd.csv.gz")
    print(f"Diagnoses: {len(diagnoses)} records")
    
    return patients, admissions, icustays, diagnoses

def identify_immunocompromised_codes():
    """Define ICD codes for immunocompromised conditions"""
    
    # These are the main categories from the paper
    immunocompromised_codes = {
        'cancer_codes': [
            # Malignant neoplasms (C00-C97)
            'C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09',
            'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
            'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32',
            'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C44', 'C45',
            'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55',
            'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66',
            'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76',
            'C77', 'C78', 'C79', 'C80', 'C81', 'C82', 'C83', 'C84', 'C85', 'C86',
            'C88', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95', 'C96', 'C97'
        ],
        'hiv_codes': ['B20', 'B21', 'B22', 'B23', 'B24'],
        'transplant_codes': ['Z94', 'T86'],  # History of transplant, transplant complications
        'immunodeficiency_codes': ['D80', 'D81', 'D82', 'D83', 'D84']
    }
    
    return immunocompromised_codes

def find_immunocompromised_patients(diagnoses):
    """Find patients with immunocompromised conditions"""
    
    codes = identify_immunocompromised_codes()
    all_codes = []
    
    # Flatten all code lists
    for category, code_list in codes.items():
        all_codes.extend(code_list)
    
    print(f"Looking for {len(all_codes)} different immunocompromised condition codes...")
    
    # Find patients with any of these codes
    # Use str.startswith to catch all subcodes (e.g., C50.1, C50.2, etc.)
    immunocompromised_patients = set()
    
    for code in all_codes:
        matching_diagnoses = diagnoses[diagnoses['icd_code'].str.startswith(code, na=False)]
        if len(matching_diagnoses) > 0:
            patients_with_code = set(matching_diagnoses['subject_id'].unique())
            immunocompromised_patients.update(patients_with_code)
            print(f"  {code}: {len(patients_with_code)} patients")
    
    return list(immunocompromised_patients)

def apply_inclusion_criteria(patients, admissions, icustays, immunocompromised_patients):
    """Apply the study inclusion criteria"""
    
    print("\n=== Applying Inclusion Criteria ===")
    
    # Start with immunocompromised patients
    cohort = pd.DataFrame({'subject_id': immunocompromised_patients})
    print(f"Immunocompromised patients: {len(cohort)}")
    
    # Add patient demographics
    cohort = cohort.merge(patients, on='subject_id', how='left')
    
    # Apply age filter (≥18 years)
    # Note: anchor_age is the patient's age at anchor_year_group
    cohort = cohort[cohort['anchor_age'] >= 18]
    print(f"After age filter (≥18): {len(cohort)}")
    
    # Get ICU stays for these patients
    patient_icus = icustays[icustays['subject_id'].isin(cohort['subject_id'])]
    
    # Apply ICU length of stay filter (≥6 hours = 0.25 days)
    patient_icus = patient_icus[patient_icus['los'] >= 0.25]
    print(f"ICU stays ≥6 hours: {len(patient_icus)}")
    
    # Keep only first ICU stay per hospital admission
    patient_icus = patient_icus.sort_values(['subject_id', 'hadm_id', 'intime'])
    patient_icus = patient_icus.groupby(['subject_id', 'hadm_id']).first().reset_index()
    print(f"First ICU stay per admission: {len(patient_icus)}")
    
    # Merge back with patient data
    final_cohort = patient_icus.merge(cohort[['subject_id', 'gender', 'anchor_age']], 
                                      on='subject_id', how='left')
    
    print(f"\n Final cohort size: {len(final_cohort)} patients")
    
    return final_cohort

def main():
    """Main function to run the patient identification"""
    
    print("=== MIMIC-IV Immunocompromised Patient Identification ===\n")
    
    # Load data
    patients, admissions, icustays, diagnoses = load_basic_tables()
    
    # Find immunocompromised patients
    immunocompromised_patients = find_immunocompromised_patients(diagnoses)
    
    # Apply inclusion criteria
    final_cohort = apply_inclusion_criteria(patients, admissions, icustays, 
                                           immunocompromised_patients)
    
    # Save results
    output_file = "../results/study_cohort.csv"
    final_cohort.to_csv(output_file, index=False)
    print(f"\n Saved cohort to: {output_file}")
    
    # Show basic statistics
    print("\n=== Cohort Statistics ===")
    print(f"Age: {final_cohort['anchor_age'].describe()}")
    print(f"Gender distribution:\n{final_cohort['gender'].value_counts()}")
    print(f"ICU length of stay: {final_cohort['los'].describe()}")

if __name__ == "__main__":
    main()
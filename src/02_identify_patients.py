"""
Step 2: Identify Immunocompromised Patients
This script finds patients who meet our study criteria with exact ICD codes from the paper
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
    
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    print(f"Date range: {icustays['intime'].min()} to {icustays['intime'].max()}")
    print(f"Year range: {icustays['intime'].dt.year.min()} to {icustays['intime'].dt.year.max()}")

    # Load diagnoses
    diagnoses = pd.read_csv(f"{data_path}/hosp/diagnoses_icd.csv.gz")
    print(f"Diagnoses: {len(diagnoses)} records")
    
    return patients, admissions, icustays, diagnoses

def main():
    """Main function to run the patient identification"""
    
    print("=== MIMIC-IV Immunocompromised Patient Identification ===\n")
    
    # Load data
    patients, admissions, icustays, diagnoses = load_basic_tables()
    
    # ADD THIS DIAGNOSTIC SECTION HERE:
    print("\n=== DIAGNOSTIC INFO ===")
    print(f"Total ICU stays in database: {len(icustays)}")
    print(f"Total unique patients in ICU: {icustays['subject_id'].nunique()}")
    print(f"Total hospital admissions with ICU: {icustays['hadm_id'].nunique()}")
    
    # Find immunocompromised patients (NOW WITH ICU TIMING FIX!)
    immunocompromised_patients = find_immunocompromised_patients(diagnoses, icustays)
    
    # ... rest of code

def identify_immunocompromised_codes():
    """Define ICD codes for immunocompromised conditions - EXACT codes from paper"""
    
    immunocompromised_codes = {
        # ICD-10 codes
        'antibody_deficiency_icd10': ['D800', 'D801', 'D802', 'D803', 'D804', 'D805', 
                                       'D830', 'D831', 'D832', 'D838', 'D839'],
        
        'cellular_deficiency_icd10': ['D821', 'D820', 'D823'],
        
        'combined_deficiency_icd10': ['D810', 'D811', 'D812', 'D813', 'D815', 'D816', 
                                       'D817', 'D8189', 'D819'],
        
        'phagocytic_defects_icd10': ['D700', 'D703', 'D71'],
        
        'complement_defects_icd10': ['D841', 'D848', 'D849'],
        
        'malignant_cancer_icd10': ['C9200', 'C9201', 'C9210', 'C9211', 'C9590', 'C9110',
                                    'C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 
                                    'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 
                                    'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 
                                    'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 
                                    'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46', 
                                    'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 
                                    'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 
                                    'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 
                                    'C72', 'C73', 'C74', 'C75', 'C76', 'C81', 'C82', 'C83', 
                                    'C84', 'C85', 'C88', 'C90', 'C91', 'C92', 'C93', 'C94', 
                                    'C95', 'C96', 'C97'],
        
        'solid_tumors_icd10': ['C800', 'C7951', 'C7952', 'C77', 'C78', 'C79', 'C80'],
        
        'solid_organ_transplant_icd10': ['Z940', 'Z941', 'Z942', 'Z944'],
        
        'immunosuppressive_therapy_icd10': ['Z79899', 'Z7901'],
        
        'hsct_icd10': ['Z9481', 'Z9482'],
        
        'hiv_icd10': ['B20', 'R75', 'B21', 'B22', 'B24'],
        
        # ICD-9 codes
        'antibody_deficiency_icd9': ['27900', '27903', '27904', '27905', '27906'],
        
        'cellular_deficiency_icd9': ['27911', '27912', '27913'],
        
        'combined_deficiency_icd9': ['279', '2792', '2793'],
        
        'phagocytic_defects_icd9': ['2880', '2881', '2882'],
        
        'complement_defects_icd9': ['2798', '2799'],
        
        'malignant_cancer_icd9': ['208', '204', '140', '141', '142', '143', '144', '145', 
                                   '146', '147', '148', '149', '150', '151', '152', '153', 
                                   '154', '155', '156', '157', '158', '159', '160', '161', 
                                   '162', '163', '164', '165', '166', '167', '168', '169', 
                                   '170', '171', '172', '1740', '1741', '1742', '1743', '1744', 
                                   '1745', '1746', '1747', '1748', '1749', '1750', '1751', 
                                   '1752', '1753', '1754', '1755', '1756', '1757', '1758', 
                                   '1759', '1760', '1761', '1762', '1763', '1764', '1765', 
                                   '1766', '1767', '1768', '1769', '1770', '1771', '1772', 
                                   '1773', '1774', '1775', '1776', '1777', '1778', '1779', 
                                   '1780', '1781', '1782', '1783', '1784', '1785', '1786', 
                                   '1787', '1788', '1789', '1790', '1791', '1792', '1793', 
                                   '1794', '1795', '1796', '1797', '1798', '1799', '1800', 
                                   '1801', '1802', '1803', '1804', '1805', '1806', '1807', 
                                   '1808', '1809', '1810', '1811', '1812', '1813', '1814', 
                                   '1815', '1816', '1817', '1818', '1819', '1820', '1821', 
                                   '1822', '1823', '1824', '1825', '1826', '1827', '1828', 
                                   '1829', '1830', '1831', '1832', '1833', '1834', '1835', 
                                   '1836', '1837', '1838', '1839', '1840', '1841', '1842', 
                                   '1843', '1844', '1845', '1846', '1847', '1848', '1849', 
                                   '1850', '1851', '1852', '1853', '1854', '1855', '1856', 
                                   '1857', '1858', '1859', '1860', '1861', '1862', '1863', 
                                   '1864', '1865', '1866', '1867', '1868', '1869', '1870', 
                                   '1871', '1872', '1873', '1874', '1875', '1876', '1877', 
                                   '1878', '1879', '1880', '1881', '1882', '1883', '1884', 
                                   '1885', '1886', '1887', '1888', '1889', '1890', '1891', 
                                   '1892', '1893', '1894', '1895', '1896', '1897', '1898', 
                                   '1899', '1900', '1901', '1902', '1903', '1904', '1905', 
                                   '1906', '1907', '1908', '1909', '1910', '1911', '1912', 
                                   '1913', '1914', '1915', '1916', '1917', '1918', '1919', 
                                   '1920', '1921', '1922', '1923', '1924', '1925', '1926', 
                                   '1927', '1928', '1929', '1930', '1931', '1932', '1933', 
                                   '1934', '1935', '1936', '1937', '1938', '1939', '1940', 
                                   '1941', '1942', '1943', '1944', '1945', '1946', '1947', 
                                   '1948', '1949', '1950', '1951', '1952', '1953', '1954', 
                                   '1955', '1956', '1957', '1958', '200', '201', '202', '203', 
                                   '204', '205', '206', '207', '208', '2386'],
        
        'solid_tumors_icd9': ['199', '1985', '196', '197', '198'],
        
        'solid_organ_transplant_icd9': ['V420', 'V421', 'V422', 'V427'],
        
        'immunosuppressive_therapy_icd9': ['V5869', 'V5863'],
        
        'hsct_icd9': ['V4281', 'V4282'],
        
        'hiv_icd9': ['042', '79571', '043', '044']
    }
    
    return immunocompromised_codes

def find_immunocompromised_patients(diagnoses, icustays):
    """
    Find patients with immunocompromised conditions
    KEY FIX: Only count diagnoses from the same hospital admission as the ICU stay
    """
    
    codes = identify_immunocompromised_codes()
    all_codes = []
    
    # Flatten all code lists
    for category, code_list in codes.items():
        all_codes.extend(code_list)
    
    print(f"\nLooking for immunocompromised conditions using {len(all_codes)} ICD codes...")
    
    # CRITICAL FIX: Merge diagnoses with ICU stays to only get diagnoses from same admission
    print("Filtering diagnoses to match ICU admissions...")
    diagnoses_with_icu = diagnoses.merge(
        icustays[['subject_id', 'hadm_id', 'stay_id']], 
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
    
    print(f"Diagnoses linked to ICU stays: {len(diagnoses_with_icu)}")
    
    # Find patients with any of these codes
    immunocompromised_patients = set()
    category_counts = {}
    
    for category, code_list in codes.items():
        category_patients = set()
        for code in code_list:
            matching_diagnoses = diagnoses_with_icu[
                diagnoses_with_icu['icd_code'].str.startswith(code, na=False)
            ]
            if len(matching_diagnoses) > 0:
                patients_with_code = set(matching_diagnoses['subject_id'].unique())
                category_patients.update(patients_with_code)
                immunocompromised_patients.update(patients_with_code)
        
        if len(category_patients) > 0:
            category_counts[category] = len(category_patients)
            print(f"  {category}: {len(category_patients)} patients")
    
    print(f"\nTotal unique immunocompromised patients: {len(immunocompromised_patients)}")
    
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
    
    print(f"\nFinal cohort size: {len(final_cohort)} patients")
    
    return final_cohort

def main():
    """Main function to run the patient identification"""
    
    print("=== MIMIC-IV Immunocompromised Patient Identification ===\n")
    
    # Load data
    patients, admissions, icustays, diagnoses = load_basic_tables()
    
    # Find immunocompromised patients (NOW WITH ICU TIMING FIX!)
    immunocompromised_patients = find_immunocompromised_patients(diagnoses, icustays)
    
    # Apply inclusion criteria
    final_cohort = apply_inclusion_criteria(patients, admissions, icustays, 
                                           immunocompromised_patients)
    
    # Save results
    output_file = "../results/study_cohort.csv"
    final_cohort.to_csv(output_file, index=False)
    print(f"\nSaved cohort to: {output_file}")
    
    # Show basic statistics
    print("\n=== Cohort Statistics ===")
    print(f"Total patients: {len(final_cohort)}")
    print(f"\nAge distribution:")
    print(final_cohort['anchor_age'].describe())
    print(f"\nGender distribution:")
    print(final_cohort['gender'].value_counts())
    print(f"\nICU length of stay (days):")
    print(final_cohort['los'].describe())

if __name__ == "__main__":
    main()



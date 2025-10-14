import pandas as pd

cohort = pd.read_csv("../results/study_cohort.csv")
print(f"Total rows: {len(cohort)}")
print(f"Unique subject_id: {cohort['subject_id'].nunique()}")
print(f"Unique hadm_id: {cohort['hadm_id'].nunique()}")  
print(f"Unique stay_id: {cohort['stay_id'].nunique()}")


"""
Step 1: Explore MIMIC-IV Data Structure
This script checks what data files we have available
"""
import os
import pandas as pd

def explore_data_structure():
    """Look at what files we have in our MIMIC-IV dataset"""
    
    data_path = "../data/mimic-iv-3.1"
    
    print("=== MIMIC-IV Data Structure ===\n")
    
    # Check if data folder exists
    if not os.path.exists(data_path):
        print(f"Data folder not found at: {data_path}")
        print("Make sure your MIMIC-IV data is in the right location!")
        return
    
    # Look at each subfolder
    subfolders = ['hosp', 'icu']
    
    for folder in subfolders:
        folder_path = os.path.join(data_path, folder)
        
        if os.path.exists(folder_path):
            print(f"{folder}/ folder contents:")
            files = os.listdir(folder_path)
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    # Get file size
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                    print(f"   📄 {file} ({size_mb:.1f} MB)")
            print()
        else:
            print(f"{folder}/ folder not found")
    
    print("=== Testing Data Loading ===\n")
    
    # Try to load a small file first
    try:
        patients_file = os.path.join(data_path, "hosp", "patients.csv.gz")
        if os.path.exists(patients_file):
            print("Loading patients.csv.gz...")
            df = pd.read_csv(patients_file)
            print(f"Successfully loaded! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First 3 rows:\n{df.head(3)}")
        else:
            print("patients.csv.gz not found")
            
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    explore_data_structure()
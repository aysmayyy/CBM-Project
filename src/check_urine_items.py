import pandas as pd

# Load d_items to see what these codes mean
d_items = pd.read_csv("../data/mimic-iv-3.1/icu/d_items.csv.gz")

# Find urine-related items
urine_items = d_items[d_items['label'].str.contains('urine|Urine', case=False, na=False)]
print("Urine-related items:")
print(urine_items[['itemid', 'label', 'category']])


import pandas as pd
data = pd.read_csv("../results/complete_ml_features.csv")
print(data.select_dtypes(include=['object']).columns.tolist())
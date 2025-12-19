#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("../results/data/complete_ml_features.csv")
print(data.select_dtypes(include=['object']).columns.tolist())



# In[ ]:





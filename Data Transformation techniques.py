import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample dataset
data = {'Values': [10, 20, 30, 40, 50, 60, 70, 80]}
df = pd.DataFrame(data)

# -----------------------------
# Normalization (Min-Max Scaling: 0 to 1)
# -----------------------------
min_max_scaler = MinMaxScaler()
df['Normalized'] = min_max_scaler.fit_transform(df[['Values']])

# -----------------------------
# Scaling (Standardization: mean = 0, std = 1)
# -----------------------------
standard_scaler = StandardScaler()
df['Standardized'] = standard_scaler.fit_transform(df[['Values']])

# -----------------------------
# Binning / Discretization
# -----------------------------

# Equal-width binning (3 bins)
df['Bins'] = pd.cut(df['Values'], bins=3, labels=["Low", "Medium", "High"])

# Equal-frequency binning (quantile-based)
df['Quantile_Bins'] = pd.qcut(df['Values'], q=3, labels=["Low", "Medium", "High"])

# Display results
print(df)

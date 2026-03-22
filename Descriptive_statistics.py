import numpy as np
import pandas as pd
from scipy import stats

# Sample dataset (you can replace this with your own data)
data = [10, 20, 20, 30, 40, 50, 60, 70, 80]

# Convert list to pandas Series for easy computation
data_series = pd.Series(data)

# Mean (Average)
mean_value = data_series.mean()

# Median (Middle value)
median_value = data_series.median()

# Mode (Most frequent value)
mode_value = data_series.mode()[0]

# Variance (Measure of spread)
variance_value = data_series.var()

# Standard Deviation (Square root of variance)
std_deviation = data_series.std()

# Quartiles (Q1, Q2, Q3)
q1 = data_series.quantile(0.25)
q2 = data_series.quantile(0.50)  # same as median
q3 = data_series.quantile(0.75)

# Range (Max - Min)
range_value = data_series.max() - data_series.min()

# Display results
print("Dataset:", data)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Variance:", variance_value)
print("Standard Deviation:", std_deviation)
print("Q1 (25%):", q1)
print("Q2 (50% / Median):", q2)
print("Q3 (75%):", q3)
print("Range:", range_value)

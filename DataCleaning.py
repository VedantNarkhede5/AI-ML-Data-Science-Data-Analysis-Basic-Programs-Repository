import pandas as pd

# -----------------------------
# STEP 1: Load CSV Files
# -----------------------------
df1 = pd.read_csv("data1.csv")
df2 = pd.read_csv("data2.csv")

print("Original Data1:\n", df1)
print("Original Data2:\n", df2)


# -----------------------------
# STEP 2: Handle Missing Values
# -----------------------------

# Check missing values
print("\nMissing values in Data1:\n", df1.isnull().sum())

# Fill missing values (example)
df1['Age'] = df1['Age'].fillna(df1['Age'].mean())  # Fill with mean
df1['City'] = df1['City'].fillna("Unknown")        # Fill with constant

# Drop rows with missing values (optional)
df1 = df1.dropna()


# -----------------------------
# STEP 3: Remove Duplicates
# -----------------------------
df1 = df1.drop_duplicates()

# Reset index after cleaning
df1 = df1.reset_index(drop=True)


# -----------------------------
# STEP 4: Data Transformation
# -----------------------------

# Convert column types
df1['Age'] = df1['Age'].astype(int)

# Standardize text (lowercase)
df1['Name'] = df1['Name'].str.lower()

# Rename columns (optional)
df1.rename(columns={'Name': 'name'}, inplace=True)


# -----------------------------
# STEP 5: Merge / Integrate Data
# -----------------------------

# Merge on common column (example: 'id')
merged_df = pd.merge(df1, df2, on='id', how='inner')

# Other merge types:
# how='left', 'right', 'outer'

print("\nMerged Data:\n", merged_df)


# -----------------------------
# STEP 6: Save Cleaned Data
# -----------------------------
merged_df.to_csv("cleaned_data.csv", index=False)

print("\nData cleaning and integration completed successfully!")

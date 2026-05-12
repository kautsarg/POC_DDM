"""
Debug normalization issue
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read data
df = pd.read_csv('data/3plex_2000_truncated.csv')
print(f"Original data shape: {df.shape}")
print(f"Original Target_cat distribution:")
print(df['Target_cat'].value_counts())

# Check for NaN values
print(f"\nNaN values in Target_cat: {df['Target_cat'].isna().sum()}")
numeric_cols = df.filter(regex=r'^\d+\.?\d*$').columns
print(f"NaN values in numeric columns: {df[numeric_cols].isna().sum().sum()}")

# Simulate dropna()
df_dropped = df.dropna()
print(f"\nAfter dropna() shape: {df_dropped.shape}")
print(f"After dropna() Target_cat distribution:")
print(df_dropped['Target_cat'].value_counts())

# Check normalization
test_X_df = df_dropped.filter(regex=r'^\d+\.?\d*$')
test_Y_df = df_dropped['Target_cat'].astype(int)

print(f"\nTest X shape: {test_X_df.shape}")
print(f"Test Y shape: {test_Y_df.shape}")
print(f"Test Y unique: {test_Y_df.unique()}")
print(f"Test Y distribution:")
print(test_Y_df.value_counts())

# Apply normalization
scaler = MinMaxScaler()
test_X_normalized = pd.DataFrame(
    scaler.fit_transform(test_X_df),
    columns=test_X_df.columns
)

print(f"\nAfter normalization X shape: {test_X_normalized.shape}")
print(f"Y shape still: {test_Y_df.shape}")

# Create a small sample to check
sample_size = 50
sample_indices = np.random.choice(len(test_Y_df), sample_size, replace=False)
sample_Y = test_Y_df.iloc[sample_indices]
print(f"\nSample of {sample_size} labels:")
print(sample_Y.value_counts())








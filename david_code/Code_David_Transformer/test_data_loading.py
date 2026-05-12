import pandas as pd
import sys
import os
sys.path.append('src')

print("=== DIRECT CSV CHECK ===")
df = pd.read_csv('data/3plex_2000_truncated.csv')
print(f'Shape: {df.shape}')
print(f'Target unique: {df["Target"].unique()}')
print(f'Target_cat unique: {df["Target_cat"].unique()}')
print(f'Target_cat value counts:')
print(df['Target_cat'].value_counts())

# Check numeric columns
numeric_cols = df.filter(regex=r'^\d+\.?\d*$').columns
print(f'Numeric columns count: {len(numeric_cols)}')

print("\n=== SIMULATE DATA LOADER LOGIC ===")
src_X_df = df.filter(regex=r'\d+\.?\d*')
src_Y_df = df['Target_cat']
print(f'X shape: {src_X_df.shape}')
print(f'Y shape: {src_Y_df.shape}')
print(f'Y unique: {src_Y_df.unique()}')
print(f'Y value counts:')
print(src_Y_df.value_counts())

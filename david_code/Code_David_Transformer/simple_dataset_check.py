import sys
import os
sys.path.append('src')
from src.data.data_loader import PCRDataset
import pandas as pd

# Load data directly like the data loader does
df = pd.read_csv('data/3plex_2000_truncated.csv').dropna()
test_X_df = df.filter(regex=r'^\d+\.?\d*$')
test_Y_df = df['Target_cat'].astype(int)

print("=== DIRECT DATASET CREATION CHECK ===")
print(f"X shape: {test_X_df.shape}")
print(f"Y shape: {test_Y_df.shape}")
print(f"Y unique: {test_Y_df.unique()}")
print(f"Y distribution: {test_Y_df.value_counts()}")

# Create dataset
test_dataset = PCRDataset(test_X_df, test_Y_df)
print(f"\nDataset length: {len(test_dataset)}")
print(f"Dataset Y unique: {test_dataset.y.unique()}")
print(f"Dataset Y distribution: {test_dataset.y.bincount()}")

print("
First 50 labels in dataset:")
print(test_dataset.y[:50])

print("
Last 50 labels in dataset:")
print(test_dataset.y[-50:])

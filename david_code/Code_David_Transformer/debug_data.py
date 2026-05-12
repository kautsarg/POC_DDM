"""
Debug script to check data loading
"""
import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root and src to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.data.data_loader import generate_data_loader

def debug_data_loading():
    print("=== DEBUG DATA LOADING ===")

    # Test data loading
    data_loaders = generate_data_loader(
        './data', 16, 16, 32,
        ['3plex_2000_truncated.csv'],
        '3plex_2000_truncated.csv',
        '3plex_2000_truncated.csv',
        'cpu', 'None'
    )

    print("\n=== CHECKING A BATCH FROM SOURCE LOADER ===")
    for batch_data, batch_labels in data_loaders['source']:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Unique labels in batch: {batch_labels.unique()}")
        print(f"Label counts: {batch_labels.bincount()}")

        # Check first sample
        print(f"First sample data shape: {batch_data[0].shape}")
        print(f"First sample label: {batch_labels[0]}")
        break

    print("\n=== CHECKING A BATCH FROM TEST LOADER ===")
    for batch_data, batch_labels in data_loaders['test']:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Unique labels in batch: {batch_labels.unique()}")
        print(f"Label counts: {batch_labels.bincount()}")
        break

    print("\n=== CHECKING RAW CSV DATA ===")
    df = pd.read_csv('./data/3plex_2000_truncated.csv')
    print(f"DataFrame shape: {df.shape}")
    print(f"Target column unique values: {df['Target'].unique()}")
    print(f"Target_cat column unique values: {df['Target_cat'].unique()}")
    print(f"Target_cat value counts:\n{df['Target_cat'].value_counts()}")

    # Check numeric columns
    numeric_cols = df.filter(regex=r'^\d+\.?\d*$').columns
    print(f"Number of numeric columns: {len(numeric_cols)}")
    print(f"First 5 numeric columns: {list(numeric_cols[:5])}")
    print(f"Last 5 numeric columns: {list(numeric_cols[-5:])}")

if __name__ == "__main__":
    debug_data_loading()








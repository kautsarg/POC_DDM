"""
Data loading utilities for PCR amplification curve classification.

This module handles:
- Loading CSV files
- Creating PyTorch datasets
- Creating data loaders for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold


class PCRDataset(Dataset):
    """PyTorch Dataset for PCR amplification curves."""
    
    def __init__(self, X, Y, dtype=torch.float32):
        """
        Initialize dataset.
        
        Args:
            X: DataFrame or numpy array of features (curves)
            Y: Series or numpy array of labels
            dtype: Data type for tensors
        """
        if isinstance(X, pd.DataFrame):
            self.X = torch.from_numpy(X.values).to(dtype=dtype)
        else:
            self.X = torch.from_numpy(np.asarray(X)).to(dtype=dtype)
            
        if isinstance(Y, pd.Series):
            self.y = torch.from_numpy(Y.values).long()
        else:
            self.y = torch.from_numpy(np.asarray(Y)).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def generate_data_loader(dir_data, 
                         train_bs, train_bt, test_b, 
                         source_data_name, target_data_name, test_data_name, 
                         device,
                         use_normalize='None'):
    """
    Create data loaders for source, target, and test domains.
    
    Args:
        dir_data: Directory containing CSV files
        train_bs: Batch size for source domain training
        train_bt: Batch size for target domain training
        test_b: Batch size for test set
        source_data_name: List of source domain CSV filenames
        target_data_name: Target domain CSV filename
        test_data_name: Test domain CSV filename
        device: torch device (cpu/cuda)
        use_normalize: Normalization method ('min_max', 'standard', or 'None')
        
    Returns:
        Dictionary with 'source', 'target', and 'test' DataLoaders
    """
    # Load source data
    source_X_df = pd.DataFrame()
    source_Y_df = pd.DataFrame()
    source_df = pd.DataFrame()
    
    for f in source_data_name:
        print(f"Loading source file: {f}")
        src_df = pd.read_csv(os.path.join(dir_data, f))
        src_X_df = src_df.filter(regex=r'\d+\.?\d*')  # Extract numeric columns
        src_Y_df = src_df['Target_cat']
        source_X_df = pd.concat([source_X_df, src_X_df], axis=0)
        source_Y_df = pd.concat([source_Y_df, src_Y_df], axis=0)
        source_df = pd.concat([source_df, src_df], axis=0)
    
    # Reload to ensure consistency
    source_df = pd.concat([pd.read_csv(os.path.join(dir_data, f)) 
                          for f in source_data_name], axis=0)
    source_X_df = source_df.filter(regex=r'\d+\.?\d*')
    source_X_df = source_X_df.dropna(axis=1)
    source_Y_df = source_df['Target_cat'].astype(int)
    
    # Load target data
    target_df = pd.read_csv(os.path.join(dir_data, target_data_name)).dropna()
    target_X_df = target_df.filter(regex=r'\d+\.?\d*')
    target_Y_df = target_df['Target_cat'].astype(int)
    
    # Load test data
    test_df = pd.read_csv(os.path.join(dir_data, test_data_name)).dropna()
    test_X_df = test_df.filter(regex=r'\d+\.?\d*')
    test_Y_df = test_df['Target_cat'].astype(int)
    
    # Normalization
    if use_normalize == 'min_max':
        scaler = MinMaxScaler()
        source_X_df = pd.DataFrame(
            scaler.fit_transform(source_X_df), 
            columns=source_X_df.columns
        )
        target_X_df = pd.DataFrame(
            scaler.transform(target_X_df), 
            columns=target_X_df.columns
        )
        test_X_df = pd.DataFrame(
            scaler.transform(test_X_df), 
            columns=test_X_df.columns
        )
    elif use_normalize == 'standard':
        scaler = StandardScaler()
        source_X_df = pd.DataFrame(
            scaler.fit_transform(source_X_df), 
            columns=source_X_df.columns
        )
        target_X_df = pd.DataFrame(
            scaler.transform(target_X_df), 
            columns=target_X_df.columns
        )
        test_X_df = pd.DataFrame(
            scaler.transform(test_X_df), 
            columns=test_X_df.columns
        )
    elif use_normalize == 'None':
        pass  # No normalization
    else:
        raise ValueError("Unsupported normalization method. Use 'min_max', 'standard', or 'None'.")
    
    # Create datasets
    source_data = PCRDataset(source_X_df, source_Y_df)
    target_data = PCRDataset(target_X_df, target_Y_df)
    test_data = PCRDataset(test_X_df, test_Y_df)
    
    # Create data loaders
    data_loaders = {
        "source": DataLoader(source_data, batch_size=train_bs, shuffle=True, drop_last=True),
        "target": DataLoader(target_data, batch_size=train_bt, shuffle=True, drop_last=True),
        "test": DataLoader(test_data, batch_size=test_b, shuffle=True)  # Enable shuffle for test data
    }
    
    # Print dataset info
    print(f"Dataset sizes - Source: {len(source_data)}, Target: {len(target_data)}, Test: {len(test_data)}")

    return data_loaders


def generate_cv_data_loader(fold_k, n_splits, dir_data, data_filename,
                             train_bs, test_bs, use_normalize='None'):
    """
    Create train/test DataLoaders for one fold of stratified k-fold CV.

    Args:
        fold_k: Zero-indexed fold number to use as the test set (0 .. n_splits-1)
        n_splits: Total number of folds
        dir_data: Directory containing the CSV file
        data_filename: CSV filename (must contain 'Target_cat' and numeric columns)
        train_bs: Batch size for the training loader
        test_bs: Batch size for the test loader
        use_normalize: 'min_max', 'standard', or 'None'
            Scaler is fit only on the training split to prevent data leakage.

    Returns:
        dict with keys 'source' and 'test' (both are DataLoaders).
        'source' == training split; 'test' == held-out fold.
    """
    df = pd.read_csv(os.path.join(dir_data, data_filename)).dropna()
    X = df.filter(regex=r'\d+\.?\d*').dropna(axis=1)
    y = df['Target_cat'].astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    train_idx, test_idx = splits[fold_k]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    if use_normalize == 'min_max':
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    elif use_normalize == 'standard':
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    elif use_normalize != 'None':
        raise ValueError("use_normalize must be 'min_max', 'standard', or 'None'.")

    train_dataset = PCRDataset(X_train, y_train)
    test_dataset  = PCRDataset(X_test,  y_test)

    print(f"  Fold {fold_k+1}/{n_splits} — train: {len(train_dataset)}, test: {len(test_dataset)}")

    return {
        "source": DataLoader(train_dataset, batch_size=train_bs, shuffle=True,  drop_last=True),
        "test":   DataLoader(test_dataset,  batch_size=test_bs,  shuffle=False),
    }


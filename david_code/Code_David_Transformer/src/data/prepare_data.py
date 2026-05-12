"""
Data preparation module for PCR amplification curve classification.

This module handles:
- Label encoding (string to integer)
- Data normalization
- CSV file preparation
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def encode_labels(df, target_column='Target', output_column='Target_cat'):
    """
    Encode string labels to integer categories.
    
    Args:
        df: DataFrame with target column
        target_column: Name of column containing string labels
        output_column: Name of output column for integer labels
        
    Returns:
        DataFrame with encoded labels and LabelEncoder object
    """
    le = preprocessing.LabelEncoder()
    df[output_column] = le.fit_transform(df[target_column])
    
    logging.info(f"Label encoding complete. Classes: {le.classes_}")
    logging.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df, le


def prepare_data(input_file, output_file, target_column='Target', normalize_control=None):
    """
    Prepare a single CSV file for training.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        target_column: Name of target column
        normalize_control: Optional control target for normalization
    """
    logging.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Encode labels
    df, le = encode_labels(df, target_column=target_column)
    
    # Optional normalization based on control
    if normalize_control is not None:
        logging.info(f"Normalizing using control: {normalize_control}")
        N_CYCLE_AVE = 10
        df.columns = df.columns.astype(str)
        df_tmp = df.copy()
        col_cycles = [x for x in df_tmp.columns if x.replace('.', '').isdigit()]
        
        df_ctrl = df_tmp[df_tmp[target_column].isin([normalize_control])].copy()
        df_ctrl['last_n_cycles_avg'] = (
            df_ctrl.filter(regex=r'\d+\.?\d*').iloc[:, -N_CYCLE_AVE:].astype(float).mean(axis=1)
        )
        ctrl_median = np.median(df_ctrl['last_n_cycles_avg'])
        df_tmp.loc[:, col_cycles] = df_tmp.loc[:, col_cycles].divide(ctrl_median)
        df = df_tmp.copy()
    
    # Save prepared data
    df.to_csv(output_file, index=False)
    logging.info(f"Prepared data saved to {output_file}")
    
    return df, le


def prepare_directory(input_dir, output_dir, target_column='Target'):
    """
    Prepare all CSV files in a directory.
    
    Args:
        input_dir: Input directory containing CSV files
        output_dir: Output directory for prepared files
        target_column: Name of target column
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logging.warning(f"No CSV files found in {input_dir}")
        return
    
    label_encoders = {}
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)
        
        logging.info(f"Processing {csv_file}...")
        df, le = prepare_data(input_path, output_path, target_column=target_column)
        label_encoders[csv_file] = le
    
    logging.info(f"All files processed. Prepared files saved to {output_dir}")
    return label_encoders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare PCR data for training')
    parser.add_argument('--data_dir', type=str, required=True, help='Input data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--target_column', type=str, default='Target', help='Target column name')
    
    args = parser.parse_args()
    
    prepare_directory(args.data_dir, args.output_dir, args.target_column)


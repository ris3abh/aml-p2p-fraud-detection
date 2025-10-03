"""Utility functions for loading and basic validation of PaySim dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

def load_paysim_data(file_path='../data/raw/PS_20174392719_1491204439457_log.csv'):
    """
    Load PaySim dataset with appropriate data types.
    
    Parameters:
    -----------
    file_path : str
        Path to the PaySim CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded PaySim dataframe
    """
    # Define column dtypes for efficient loading
    dtypes = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'object',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }
    
    # Load data
    df = pd.read_csv(file_path, dtype=dtypes)
    
    # Basic validation
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"\nFraud rate: {df['isFraud'].mean():.4%}")
    print(f"Flagged fraud rate: {df['isFlaggedFraud'].mean():.4%}")
    
    return df

def get_basic_info(df):
    """Get basic information about the dataset."""
    info = {
        'n_transactions': len(df),
        'n_unique_orig': df['nameOrig'].nunique(),
        'n_unique_dest': df['nameDest'].nunique(),
        'n_fraud': df['isFraud'].sum(),
        'fraud_rate': df['isFraud'].mean(),
        'time_span_hours': df['step'].max(),
        'transaction_types': df['type'].value_counts().to_dict()
    }
    return info

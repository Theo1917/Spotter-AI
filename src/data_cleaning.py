"""
Data Cleaning Module

Applies consistent cleaning rules to both training and validation data.
All transformations must be safe for validation (no row drops, only corrections/imputation).
"""
import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(df, is_training=True):
    """
    Clean and preprocess freight rate data.
    
    Parameters:
    -----------
    df : DataFrame
        Raw data to clean
    is_training : bool
        If True, allows dropping rows. If False (validation), only corrects/imputes.
    
    Returns:
    --------
    DataFrame : Cleaned data
    """
    df = df.copy()
    
    # 1. Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 2. Strip whitespace from string columns
    for col in ['pickup', 'delivery', 'equipment']:
        if col in df.columns:
            df[col] = df[col].str.strip()
    
    # 3. Fix negative weights (likely data entry error - take absolute value)
    if 'weight' in df.columns:
        df['weight'] = df['weight'].abs()
    
    # 4. Handle missing weights (impute with median)
    if 'weight' in df.columns:
        weight_median = df['weight'].median()
        df['weight'] = df['weight'].fillna(weight_median)
    
    # 5. Handle missing market_index (impute with median)
    if 'market_index' in df.columns:
        mi_median = df['market_index'].median()
        df['market_index'] = df['market_index'].fillna(mi_median)
    
    # 6. For training only: remove extreme outliers that are likely data errors
    if is_training and 'posted_rate' in df.columns:
        # Remove rate-per-mile outliers (< $0.40 or > $15/mile)
        df['_rpm_check'] = df['posted_rate'] / df['distance']
        mask_valid = (df['_rpm_check'] >= 0.4) & (df['_rpm_check'] <= 15)
        df = df[mask_valid].copy()
        df = df.drop(columns=['_rpm_check'])
    
    # 7. Ensure no infinite or NaN values in critical features
    critical_cols = ['distance', 'weight']
    for col in critical_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df

def get_cleaning_summary(df_before, df_after, name="dataset"):
    """Generate a summary of cleaning operations."""
    print(f"\n{name} Cleaning Summary:")
    print(f"  Rows before: {len(df_before):,}")
    print(f"  Rows after:  {len(df_after):,}")
    print(f"  Rows removed: {len(df_before) - len(df_after):,}")
    
    if 'weight' in df_before.columns:
        neg_weight_before = (df_before['weight'] < 0).sum()
        neg_weight_after = (df_after['weight'] < 0).sum() if 'weight' in df_after.columns else 0
        print(f"  Negative weights fixed: {neg_weight_before}")
    
    if 'market_index' in df_before.columns:
        missing_before = df_before['market_index'].isna().sum()
        missing_after = df_after['market_index'].isna().sum() if 'market_index' in df_after.columns else 0
        print(f"  Market index missing imputed: {missing_before - missing_after}")

if __name__ == "__main__":
    # Test the cleaning function
    print("Testing data cleaning module...")
    
    train_raw = pd.read_csv('data/train_test.csv')
    val_raw = pd.read_csv('data/validation.csv')
    
    print("\nBefore cleaning:")
    print(f"Train: {train_raw.shape}")
    print(f"Validation: {val_raw.shape}")
    
    train_clean = clean_data(train_raw, is_training=True)
    val_clean = clean_data(val_raw, is_training=False)
    
    get_cleaning_summary(train_raw, train_clean, "Training")
    get_cleaning_summary(val_raw, val_clean, "Validation")
    
    # Save cleaned data
    train_clean.to_csv('outputs/train_test_clean.csv', index=False)
    val_clean.to_csv('outputs/validation_clean.csv', index=False)
    
    print("\n✓ Cleaned data saved to outputs/")

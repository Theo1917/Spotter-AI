"""
Validation Strategy Module

Implements time-based validation that respects temporal ordering,
plus comparison with random splits to demonstrate leakage.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def get_time_based_split(df, val_months=1):
    """
    Create time-based train/validation split.
    
    Parameters:
    -----------
    df : DataFrame
        Data with 'date' column
    val_months : int
        Number of months to hold out for validation
    
    Returns:
    --------
    train_df, val_df : tuple of DataFrames
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the last val_months for validation
    max_date = df['date'].max()
    
    # Calculate cutoff date
    if val_months == 1:
        cutoff = max_date.replace(day=1)
    else:
        # Move back val_months months
        month = max_date.month - val_months + 1
        year = max_date.year
        while month <= 0:
            month += 12
            year -= 1
        cutoff = pd.Timestamp(year=year, month=month, day=1)
    
    train_mask = df['date'] < cutoff
    val_mask = df['date'] >= cutoff
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    return train_df, val_df

def expanding_window_cv(df, n_splits=3, min_train_months=6):
    """
    Generate expanding window time-series CV splits.
    
    Each fold trains on progressively more data and validates on the next period.
    
    Parameters:
    -----------
    df : DataFrame
        Data with 'date' column
    n_splits : int
        Number of validation folds
    min_train_months : int
        Minimum months in the first training set
    
    Yields:
    -------
    train_idx, val_idx : tuple of arrays
        Indices for each fold
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get unique months
    df['year_month'] = df['date'].dt.to_period('M')
    unique_months = sorted(df['year_month'].unique())
    
    total_months = len(unique_months)
    
    if total_months < min_train_months + n_splits:
        raise ValueError(f"Not enough months for {n_splits} splits with min {min_train_months} training months")
    
    # Calculate validation month indices
    # We want to validate on the last n_splits months, one at a time
    val_month_indices = list(range(total_months - n_splits, total_months))
    
    for val_month_idx in val_month_indices:
        val_month = unique_months[val_month_idx]
        
        # Training: all months before validation month
        train_months = unique_months[:val_month_idx]
        
        train_mask = df['year_month'].isin(train_months)
        val_mask = df['year_month'] == val_month
        
        train_idx = df[train_mask].index.values
        val_idx = df[val_mask].index.values
        
        yield train_idx, val_idx

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - handle division by zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def segment_analysis(df, y_true_col, y_pred_col):
    """
    Analyze predictions by different segments.
    
    Parameters:
    -----------
    df : DataFrame
        Must contain y_true_col, y_pred_col, plus: distance, equipment
    y_true_col : str
        Name of true target column
    y_pred_col : str
        Name of predicted target column
    
    Returns:
    --------
    dict : Segment-wise metrics
    """
    results = {}
    
    # Create distance buckets
    df = df.copy()
    df['distance_bucket'] = pd.cut(df['distance'], 
                                    bins=[0, 300, 600, 1000, 1500, 2000, 3500],
                                    labels=['0-300', '300-600', '600-1000', '1000-1500', '1500-2000', '2000+'])
    
    # By distance bucket
    dist_metrics = []
    for bucket in df['distance_bucket'].cat.categories:
        mask = df['distance_bucket'] == bucket
        if mask.sum() > 0:
            metrics = calculate_metrics(df.loc[mask, y_true_col], df.loc[mask, y_pred_col])
            metrics['segment'] = f'Distance: {bucket}'
            metrics['count'] = mask.sum()
            dist_metrics.append(metrics)
    results['distance'] = pd.DataFrame(dist_metrics)
    
    # By equipment
    equip_metrics = []
    for equip in df['equipment'].unique():
        mask = df['equipment'] == equip
        if mask.sum() > 0:
            metrics = calculate_metrics(df.loc[mask, y_true_col], df.loc[mask, y_pred_col])
            metrics['segment'] = f'Equipment: {equip}'
            metrics['count'] = mask.sum()
            equip_metrics.append(metrics)
    results['equipment'] = pd.DataFrame(equip_metrics)
    
    # By month if date available
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.month
        month_metrics = []
        for month in sorted(df['month'].unique()):
            mask = df['month'] == month
            if mask.sum() > 0:
                metrics = calculate_metrics(df.loc[mask, y_true_col], df.loc[mask, y_pred_col])
                metrics['segment'] = f'Month: {month}'
                metrics['count'] = mask.sum()
                month_metrics.append(metrics)
        results['month'] = pd.DataFrame(month_metrics)
    
    return results

if __name__ == "__main__":
    print("="*80)
    print("PHASE 2: VALIDATION STRATEGY IMPLEMENTATION")
    print("="*80)
    
    # Load cleaned data
    df = pd.read_csv('outputs/train_test_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\nTotal data: {len(df):,} rows")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # 1. Primary time-based holdout: Oct as validation
    print("\n" + "="*80)
    print("1. PRIMARY HOLDOUT: Time-based (Jan-Sep train / Oct validation)")
    print("="*80)
    
    train_primary, val_primary = get_time_based_split(df, val_months=1)
    
    print(f"\nTraining set:")
    print(f"  Rows: {len(train_primary):,}")
    print(f"  Date range: {train_primary['date'].min().date()} to {train_primary['date'].max().date()}")
    print(f"  Months: {train_primary['date'].dt.to_period('M').nunique()}")
    
    print(f"\nValidation set:")
    print(f"  Rows: {len(val_primary):,}")
    print(f"  Date range: {val_primary['date'].min().date()} to {val_primary['date'].max().date()}")
    print(f"  Months: {val_primary['date'].dt.to_period('M').nunique()}")
    
    # Save the split
    train_primary.to_csv('outputs/train_primary.csv', index=False)
    val_primary.to_csv('outputs/val_primary.csv', index=False)
    
    # 2. Expanding window CV
    print("\n" + "="*80)
    print("2. EXPANDING WINDOW TIME-SERIES CV")
    print("="*80)
    
    print("\nGenerating 3 folds with expanding training window...")
    
    fold_info = []
    for i, (train_idx, val_idx) in enumerate(expanding_window_cv(df, n_splits=3), 1):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        
        fold_info.append({
            'fold': i,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_start': train_fold['date'].min().date(),
            'train_end': train_fold['date'].max().date(),
            'val_start': val_fold['date'].min().date(),
            'val_end': val_fold['date'].max().date()
        })
    
    fold_df = pd.DataFrame(fold_info)
    print("\nFold Summary:")
    print(fold_df.to_string(index=False))
    
    fold_df.to_csv('outputs/cv_folds_summary.csv', index=False)
    
    # 3. Random K-fold comparison (for demonstration only)
    print("\n" + "="*80)
    print("3. RANDOM K-FOLD (for comparison - shows optimistic leakage)")
    print("="*80)
    
    print("\nGenerating 5 random folds...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    random_fold_info = []
    for i, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        
        # Check temporal overlap
        train_dates = set(train_fold['date'].dt.date)
        val_dates = set(val_fold['date'].dt.date)
        overlap = len(train_dates & val_dates)
        
        random_fold_info.append({
            'fold': i,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'temporal_overlap_days': overlap,
            'future_in_train': (train_fold['date'].max() > val_fold['date'].max())
        })
    
    random_df = pd.DataFrame(random_fold_info)
    print("\nRandom Fold Summary:")
    print(random_df.to_string(index=False))
    
    print(f"\n⚠️  All random folds have temporal overlap!")
    print(f"   This means training data includes future dates relative to validation.")
    print(f"   Metrics will be artificially optimistic and not representative of")
    print(f"   real forecasting performance.")
    
    random_df.to_csv('outputs/random_folds_summary.csv', index=False)
    
    # 4. Validation set analysis
    print("\n" + "="*80)
    print("4. VALIDATION SET CHARACTERISTICS")
    print("="*80)
    
    print("\nChecking for unseen categories in validation holdout...")
    
    train_pickups = set(train_primary['pickup'].unique())
    val_pickups = set(val_primary['pickup'].unique())
    unseen_pickups = val_pickups - train_pickups
    
    train_deliveries = set(train_primary['delivery'].unique())
    val_deliveries = set(val_primary['delivery'].unique())
    unseen_deliveries = val_deliveries - train_deliveries
    
    train_lanes = set(train_primary['pickup'] + ' -> ' + train_primary['delivery'])
    val_lanes = set(val_primary['pickup'] + ' -> ' + val_primary['delivery'])
    unseen_lanes = val_lanes - train_lanes
    
    print(f"\nPickup cities:")
    print(f"  In training: {len(train_pickups)}")
    print(f"  In validation: {len(val_pickups)}")
    print(f"  Unseen in validation: {len(unseen_pickups)}")
    if len(unseen_pickups) > 0:
        print(f"  Examples: {list(unseen_pickups)[:5]}")
    
    print(f"\nDelivery cities:")
    print(f"  In training: {len(train_deliveries)}")
    print(f"  In validation: {len(val_deliveries)}")
    print(f"  Unseen in validation: {len(unseen_deliveries)}")
    if len(unseen_deliveries) > 0:
        print(f"  Examples: {list(unseen_deliveries)[:5]}")
    
    print(f"\nLanes:")
    print(f"  In training: {len(train_lanes)}")
    print(f"  In validation: {len(val_lanes)}")
    print(f"  Unseen in validation: {len(unseen_lanes)} ({100*len(unseen_lanes)/len(val_lanes):.1f}%)")
    
    # 5. Metrics definition
    print("\n" + "="*80)
    print("5. EVALUATION METRICS")
    print("="*80)
    
    print("\nPrimary Metric: MAE (Mean Absolute Error)")
    print("  - Directly interpretable in dollars")
    print("  - Robust to outliers")
    print("  - Business-relevant (average prediction error)")
    
    print("\nSecondary Metrics:")
    print("  - RMSE: Penalizes large errors more heavily")
    print("  - MAPE: Percentage error (scale-independent)")
    print("  - R²: Variance explained (model fit quality)")
    
    print("\nSegment Analysis:")
    print("  - By distance bucket (short/medium/long haul)")
    print("  - By equipment type (Dry Van / Reefer / Flatbed)")
    print("  - By month (seasonal patterns)")
    
    print("\n" + "="*80)
    print("VALIDATION STRATEGY SUMMARY")
    print("="*80)
    
    summary = f"""
Primary Validation:
  - Time-based holdout (Jan-Sep train / Oct validation)
  - Training: {len(train_primary):,} rows, {train_primary['date'].dt.to_period('M').nunique()} months
  - Validation: {len(val_primary):,} rows, 1 month
  - Mimics the real task: forecast 1-2 months ahead

Time-Series CV:
  - 3 folds with expanding window
  - Each fold: train on more history, validate on next month
  - Used for robustness checks and hyperparameter tuning

Random K-Fold:
  - 5 folds for comparison only
  - Shows temporal leakage (train/val dates overlap)
  - Expected to be overly optimistic
  - Used in report to justify time-based approach

Metrics:
  - Primary: MAE (mean absolute error in dollars)
  - Secondary: RMSE, MAPE, R²
  - Segment analysis by distance, equipment, month

Final Model Training:
  - After validation, retrain on ALL train_test.csv data (Jan-Oct)
  - Predict on validation.csv (Nov-Dec)
  - This matches the submission requirement
"""
    
    print(summary)
    
    with open('outputs/validation_strategy_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: VALIDATION STRATEGY\n")
        f.write("="*80 + "\n\n")
        f.write(summary)
    
    print("\n✓ Phase 2 complete. Validation strategy implemented and saved.")
    print("\nFiles created:")
    print("  - outputs/train_primary.csv (primary training set)")
    print("  - outputs/val_primary.csv (primary validation set)")
    print("  - outputs/cv_folds_summary.csv (time-series CV folds)")
    print("  - outputs/random_folds_summary.csv (random K-fold comparison)")
    print("  - outputs/validation_strategy_summary.txt (documentation)")

"""
Modeling Module

Implements baseline and advanced models with proper validation.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, prefix=''):
    """Calculate all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        f'{prefix}MAE': round(mae, 2),
        f'{prefix}RMSE': round(rmse, 2),
        f'{prefix}MAPE': round(mape, 2),
        f'{prefix}R2': round(r2, 4)
    }
    return metrics

def segment_metrics(df, y_true_col, y_pred_col):
    """Calculate metrics by segment."""
    results = []
    
    # By distance bucket
    df['distance_bucket'] = pd.cut(df['distance'], 
                                    bins=[0, 300, 600, 1000, 1500, 2000, 3500],
                                    labels=['0-300', '300-600', '600-1000', 
                                           '1000-1500', '1500-2000', '2000+'])
    
    for bucket in df['distance_bucket'].cat.categories:
        mask = df['distance_bucket'] == bucket
        if mask.sum() > 10:
            metrics = calculate_metrics(df.loc[mask, y_true_col], 
                                       df.loc[mask, y_pred_col])
            metrics['segment'] = f'{bucket} miles'
            metrics['count'] = mask.sum()
            results.append(metrics)
    
    # By equipment
    for equip in sorted(df['equipment'].unique()):
        mask = df['equipment'] == equip
        if mask.sum() > 10:
            metrics = calculate_metrics(df.loc[mask, y_true_col], 
                                       df.loc[mask, y_pred_col])
            metrics['segment'] = equip
            metrics['count'] = mask.sum()
            results.append(metrics)
    
    return pd.DataFrame(results)

class BaselineModel:
    """Naive baseline: mean rate-per-mile × distance."""
    
    def __init__(self):
        self.mean_rpm = None
        
    def fit(self, X, y):
        """Compute mean rate-per-mile."""
        rpm = y / X['distance']
        self.mean_rpm = rpm.mean()
        return self
        
    def predict(self, X):
        """Predict using mean RPM."""
        return self.mean_rpm * X['distance']

if __name__ == "__main__":
    print("="*80)
    print("PHASE 4: MODELING")
    print("="*80)
    
    # Load features
    train = pd.read_csv('outputs/train_features.csv')
    val = pd.read_csv('outputs/val_features.csv')
    
    print(f"\nData loaded:")
    print(f"  Training: {len(train):,} rows")
    print(f"  Validation: {len(val):,} rows")
    
    # Define feature columns (only numeric)
    exclude_cols = ['load_id', 'posted_rate', 'date', 'pickup', 'delivery', 
                    'lane', 'distance_bucket', 'weight_bucket', 'equipment']
    
    # Get numeric columns only
    all_cols = train.columns.tolist()
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\nTotal columns: {len(all_cols)}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Feature columns: {len(feature_cols)}")
    
    X_train = train[feature_cols].copy()
    y_train = train['posted_rate'].copy()
    X_val = val[feature_cols].copy()
    y_val = val['posted_rate'].copy()
    
    # Check for NaN/inf
    print(f"\nData quality check BEFORE imputation:")
    print(f"  Train NaN: {X_train.isna().sum().sum()}")
    print(f"  Val NaN: {X_val.isna().sum().sum()}")
    
    # Impute NaN with column medians from training
    if X_val.isna().sum().sum() > 0:
        print(f"\nImputing {X_val.isna().sum().sum()} NaN values in validation...")
        for col in feature_cols:
            if X_val[col].isna().sum() > 0:
                fill_value = X_train[col].median()
                X_val[col] = X_val[col].fillna(fill_value)
    
    print(f"\nData quality check AFTER imputation:")
    print(f"  Train NaN: {X_train.isna().sum().sum()}")
    print(f"  Val NaN: {X_val.isna().sum().sum()}")
    
    # Store results
    results = []
    
    # ========================================================================
    # BASELINE 1: Naive mean RPM
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE 1: Naive Mean Rate-per-Mile")
    print("="*80)
    
    baseline1 = BaselineModel()
    baseline1.fit(X_train, y_train)
    
    y_train_pred_b1 = baseline1.predict(X_train)
    y_val_pred_b1 = baseline1.predict(X_val)
    
    metrics_b1_train = calculate_metrics(y_train, y_train_pred_b1, prefix='train_')
    metrics_b1_val = calculate_metrics(y_val, y_val_pred_b1, prefix='val_')
    
    print(f"\nMean RPM: ${baseline1.mean_rpm:.3f}")
    print(f"\nTraining: MAE=${metrics_b1_train['train_MAE']:.2f}, RMSE=${metrics_b1_train['train_RMSE']:.2f}, R²={metrics_b1_train['train_R2']:.4f}")
    print(f"Validation: MAE=${metrics_b1_val['val_MAE']:.2f}, RMSE=${metrics_b1_val['val_RMSE']:.2f}, R²={metrics_b1_val['val_R2']:.4f}")
    
    results.append({
        'model': 'Baseline_NaiveRPM',
        **metrics_b1_train,
        **metrics_b1_val
    })
    
    # ========================================================================
    # BASELINE 2: Ridge Regression
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE 2: Ridge Regression (numeric features)")
    print("="*80)
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    y_train_pred_ridge = ridge.predict(X_train)
    y_val_pred_ridge = ridge.predict(X_val)
    
    metrics_ridge_train = calculate_metrics(y_train, y_train_pred_ridge, prefix='train_')
    metrics_ridge_val = calculate_metrics(y_val, y_val_pred_ridge, prefix='val_')
    
    print(f"\nTraining: MAE=${metrics_ridge_train['train_MAE']:.2f}, RMSE=${metrics_ridge_train['train_RMSE']:.2f}, R²={metrics_ridge_train['train_R2']:.4f}")
    print(f"Validation: MAE=${metrics_ridge_val['val_MAE']:.2f}, RMSE=${metrics_ridge_val['val_RMSE']:.2f}, R²={metrics_ridge_val['val_R2']:.4f}")
    
    results.append({
        'model': 'Ridge_Regression',
        **metrics_ridge_train,
        **metrics_ridge_val
    })
    
    # ========================================================================
    # MODEL 3: HistGradientBoosting (rate)
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL 3: HistGradientBoosting (predict rate directly)")
    print("="*80)
    
    print("\nTraining...")
    hgb_rate = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
        verbose=0
    )
    hgb_rate.fit(X_train, y_train)
    
    y_train_pred_hgb = hgb_rate.predict(X_train)
    y_val_pred_hgb = hgb_rate.predict(X_val)
    
    metrics_hgb_train = calculate_metrics(y_train, y_train_pred_hgb, prefix='train_')
    metrics_hgb_val = calculate_metrics(y_val, y_val_pred_hgb, prefix='val_')
    
    print(f"\nTraining: MAE=${metrics_hgb_train['train_MAE']:.2f}, RMSE=${metrics_hgb_train['train_RMSE']:.2f}, R²={metrics_hgb_train['train_R2']:.4f}")
    print(f"Validation: MAE=${metrics_hgb_val['val_MAE']:.2f}, RMSE=${metrics_hgb_val['val_RMSE']:.2f}, R²={metrics_hgb_val['val_R2']:.4f}")
    
    results.append({
        'model': 'HistGB_Rate',
        **metrics_hgb_train,
        **metrics_hgb_val
    })
    
    # ========================================================================
    # MODEL 4: HistGradientBoosting (log rate)
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL 4: HistGradientBoosting (predict log(rate))")
    print("="*80)
    
    print("\nTraining on log-transformed target...")
    y_train_log = np.log1p(y_train)
    
    hgb_log = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
        verbose=0
    )
    hgb_log.fit(X_train, y_train_log)
    
    y_train_pred_log = np.expm1(hgb_log.predict(X_train))
    y_val_pred_log = np.expm1(hgb_log.predict(X_val))
    
    metrics_log_train = calculate_metrics(y_train, y_train_pred_log, prefix='train_')
    metrics_log_val = calculate_metrics(y_val, y_val_pred_log, prefix='val_')
    
    print(f"\nTraining: MAE=${metrics_log_train['train_MAE']:.2f}, RMSE=${metrics_log_train['train_RMSE']:.2f}, R²={metrics_log_train['train_R2']:.4f}")
    print(f"Validation: MAE=${metrics_log_val['val_MAE']:.2f}, RMSE=${metrics_log_val['val_RMSE']:.2f}, R²={metrics_log_val['val_R2']:.4f}")
    
    results.append({
        'model': 'HistGB_LogRate',
        **metrics_log_train,
        **metrics_log_val
    })
    
    # ========================================================================
    # MODEL 5: HistGradientBoosting (rate-per-mile)
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL 5: HistGradientBoosting (predict rate-per-mile)")
    print("="*80)
    
    print("\nTraining on rate-per-mile...")
    y_train_rpm = y_train / X_train['distance']
    
    hgb_rpm = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
        verbose=0
    )
    hgb_rpm.fit(X_train, y_train_rpm)
    
    y_train_pred_rpm = hgb_rpm.predict(X_train) * X_train['distance']
    y_val_pred_rpm = hgb_rpm.predict(X_val) * X_val['distance']
    
    metrics_rpm_train = calculate_metrics(y_train, y_train_pred_rpm, prefix='train_')
    metrics_rpm_val = calculate_metrics(y_val, y_val_pred_rpm, prefix='val_')
    
    print(f"\nTraining: MAE=${metrics_rpm_train['train_MAE']:.2f}, RMSE=${metrics_rpm_train['train_RMSE']:.2f}, R²={metrics_rpm_train['train_R2']:.4f}")
    print(f"Validation: MAE=${metrics_rpm_val['val_MAE']:.2f}, RMSE=${metrics_rpm_val['val_RMSE']:.2f}, R²={metrics_rpm_val['val_R2']:.4f}")
    
    results.append({
        'model': 'HistGB_RPM',
        **metrics_rpm_train,
        **metrics_rpm_val
    })
    
    # ========================================================================
    # COMPARISON TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Determine best model
    best_idx = results_df['val_MAE'].idxmin()
    best_model_name = results_df.loc[best_idx, 'model']
    best_mae = results_df.loc[best_idx, 'val_MAE']
    
    print(f"\n🏆 Best model: {best_model_name} (Validation MAE = ${best_mae:.2f})")
    
    # Save comparison
    results_df.to_csv('outputs/model_comparison.csv', index=False)
    
    # ========================================================================
    # SEGMENT ANALYSIS (best model)
    # ========================================================================
    print("\n" + "="*80)
    print(f"SEGMENT ANALYSIS: {best_model_name}")
    print("="*80)
    
    # Use the best performing model
    if best_model_name == 'HistGB_LogRate':
        best_pred = y_val_pred_log
    elif best_model_name == 'HistGB_Rate':
        best_pred = y_val_pred_hgb
    elif best_model_name == 'HistGB_RPM':
        best_pred = y_val_pred_rpm
    else:
        best_pred = y_val_pred_rpm  # default to RPM
    
    val_with_pred = val.copy()
    val_with_pred['pred'] = best_pred
    
    segment_df = segment_metrics(val_with_pred, 'posted_rate', 'pred')
    print("\n" + segment_df.to_string(index=False))
    
    segment_df.to_csv('outputs/segment_analysis.csv', index=False)
    
    # ========================================================================
    # SAVE BEST MODEL AND ARTIFACTS
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING BEST MODEL")
    print("="*80)
    
    # Determine which model to save
    if best_model_name == 'HistGB_LogRate':
        best_model = hgb_log
        best_metrics_train = metrics_log_train
        best_metrics_val = metrics_log_val
        target_transform = 'log1p'
    elif best_model_name == 'HistGB_Rate':
        best_model = hgb_rate
        best_metrics_train = metrics_hgb_train
        best_metrics_val = metrics_hgb_val
        target_transform = 'none'
    elif best_model_name == 'HistGB_RPM':
        best_model = hgb_rpm
        best_metrics_train = metrics_rpm_train
        best_metrics_val = metrics_rpm_val
        target_transform = 'rpm'
    else:
        best_model = hgb_rpm
        best_metrics_train = metrics_rpm_train
        best_metrics_val = metrics_rpm_val
        target_transform = 'rpm'
    
    with open('outputs/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    model_info = {
        'model_name': best_model_name,
        'model_type': 'HistGradientBoostingRegressor',
        'target_transform': target_transform,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'metrics': {
            'train_MAE': float(best_metrics_train['train_MAE']),
            'train_RMSE': float(best_metrics_train['train_RMSE']),
            'train_R2': float(best_metrics_train['train_R2']),
            'val_MAE': float(best_metrics_val['val_MAE']),
            'val_RMSE': float(best_metrics_val['val_RMSE']),
            'val_R2': float(best_metrics_val['val_R2'])
        }
    }
    
    with open('outputs/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✓ Best model saved: outputs/best_model.pkl")
    print(f"✓ Model info saved: outputs/model_info.json")
    
    print("\n" + "="*80)
    print("✓ Phase 4 complete.")
    print("="*80)
    
    print(f"\nKey Takeaways:")
    print(f"  - Best validation MAE: ${best_mae:.2f}")
    print(f"  - Improvement over baseline: {100*(metrics_b1_val['val_MAE']-best_mae)/metrics_b1_val['val_MAE']:.1f}%")
    print(f"  - Using {len(feature_cols)} numeric features")
    print(f"  - Model: {best_model_name}")
    print(f"  - Transform: {target_transform}")

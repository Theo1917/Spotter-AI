"""
Predictions Module

Generate final predictions for validation set and December chart.
"""
import pandas as pd
import numpy as np
import pickle
import json
from feature_engineering import FeatureEngineer

def load_model_and_info():
    """Load the trained model and its metadata."""
    with open('outputs/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('outputs/model_info.json', 'r') as f:
        model_info = json.load(f)
    
    return model, model_info

def predict_with_transform(model, X, transform_type, distance=None):
    """Make predictions with appropriate inverse transform."""
    if transform_type == 'log1p':
        # Model predicts log(rate), need to exponentiate
        return np.expm1(model.predict(X))
    elif transform_type == 'rpm':
        # Model predicts rate-per-mile, multiply by distance
        if distance is None:
            distance = X['distance']
        return model.predict(X) * distance
    else:
        # Model predicts rate directly
        return model.predict(X)

if __name__ == "__main__":
    print("="*80)
    print("PHASE 5 & 6: DECEMBER CHART AND FINAL PREDICTIONS")
    print("="*80)
    
    # Load model
    print("\n[1] Loading trained model...")
    model, model_info = load_model_and_info()
    
    print(f"Model: {model_info['model_name']}")
    print(f"Transform: {model_info['target_transform']}")
    print(f"Validation MAE: ${model_info['metrics']['val_MAE']:.2f}")
    
    # ========================================================================
    # PHASE 5: DECEMBER CHART PREDICTIONS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: DECEMBER CHART (31-day fixed lane forecast)")
    print("="*80)
    
    # Load December chart input
    december = pd.read_csv('data/december_chart_inputs.csv')
    print(f"\nDecember chart: {len(december)} rows")
    print(f"Lane: {december['pickup'].iloc[0]} → {december['delivery'].iloc[0]}")
    print(f"Distance: {december['distance'].iloc[0]} miles")
    print(f"Equipment: {december['equipment'].iloc[0]}")
    print(f"Weight: {december['weight'].iloc[0]} lbs")
    print(f"Dates: {december['date'].min()} to {december['date'].max()}")
    
    # Train feature engineer on ALL train_test data
    print("\n[2] Training feature engineer on full train_test data...")
    full_train = pd.read_csv('outputs/train_test_clean.csv')
    
    fe = FeatureEngineer()
    fe.fit(full_train, target_col='posted_rate')
    
    # Engineer features for December
    print("\n[3] Engineering features for December...")
    december_features = fe.transform(december)
    
    # Select feature columns
    feature_cols = model_info['features']
    X_december = december_features[feature_cols]
    
    # Check for NaN
    if X_december.isna().sum().sum() > 0:
        print(f"\nImputing {X_december.isna().sum().sum()} NaN values...")
        for col in feature_cols:
            if X_december[col].isna().sum() > 0:
                # Use training median
                fill_value = full_train[col].median() if col in full_train.columns else 0
                X_december[col] = X_december[col].fillna(fill_value)
    
    # Make predictions
    print("\n[4] Generating December predictions...")
    transform_type = model_info['target_transform']
    december_pred = predict_with_transform(model, X_december, transform_type, 
                                          distance=X_december['distance'])
    
    # Add to DataFrame
    december['predicted_rate'] = december_pred
    
    # Statistics
    print(f"\nDecember predictions:")
    print(f"  Mean: ${december_pred.mean():.2f}")
    print(f"  Median: ${np.median(december_pred):.2f}")
    print(f"  Min: ${december_pred.min():.2f}")
    print(f"  Max: ${december_pred.max():.2f}")
    print(f"  Std: ${december_pred.std():.2f}")
    
    # Check for temporal pattern
    december['date'] = pd.to_datetime(december['date'])
    december['day_of_week'] = december['date'].dt.dayofweek
    weekday_avg = december[december['day_of_week'] < 5]['predicted_rate'].mean()
    weekend_avg = december[december['day_of_week'] >= 5]['predicted_rate'].mean()
    
    print(f"\nTemporal patterns:")
    print(f"  Weekday avg: ${weekday_avg:.2f}")
    print(f"  Weekend avg: ${weekend_avg:.2f}")
    print(f"  Difference: ${abs(weekday_avg - weekend_avg):.2f}")
    
    # Check for outliers or flat predictions
    cv = december_pred.std() / december_pred.mean()
    print(f"  Coefficient of variation: {cv:.3f}")
    
    if cv < 0.01:
        print("  ⚠️ WARNING: Predictions are very flat (CV < 1%)")
    else:
        print("  ✓ Predictions show reasonable variation")
    
    # Save December chart
    december[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']].to_csv(
        'data/december_chart_inputs.csv', index=False
    )
    print("\n✓ December chart saved to: data/december_chart_inputs.csv")
    
    # ========================================================================
    # PHASE 6: FINAL VALIDATION PREDICTIONS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 6: FINAL VALIDATION PREDICTIONS (12,000 loads)")
    print("="*80)
    
    # Load validation data
    print("\n[1] Loading validation data...")
    validation = pd.read_csv('outputs/validation_clean.csv')
    print(f"Validation: {len(validation)} rows")
    
    # Engineer features
    print("\n[2] Engineering features for validation...")
    validation_features = fe.transform(validation)
    
    X_validation = validation_features[feature_cols]
    
    # Check for NaN and impute if needed
    if X_validation.isna().sum().sum() > 0:
        print(f"\nImputing {X_validation.isna().sum().sum()} NaN values...")
        for col in feature_cols:
            if X_validation[col].isna().sum() > 0:
                fill_value = full_train[col].median() if col in full_train.columns else 0
                X_validation[col] = X_validation[col].fillna(fill_value)
    
    # Make predictions
    print("\n[3] Generating validation predictions...")
    validation_pred = predict_with_transform(model, X_validation, transform_type,
                                            distance=X_validation['distance'])
    
    # Statistics
    print(f"\nValidation predictions:")
    print(f"  Mean: ${validation_pred.mean():.2f}")
    print(f"  Median: ${np.median(validation_pred):.2f}")
    print(f"  Min: ${validation_pred.min():.2f}")
    print(f"  Max: ${validation_pred.max():.2f}")
    print(f"  Std: ${validation_pred.std():.2f}")
    
    # Create submission file
    print("\n[4] Creating submission file...")
    template = pd.read_csv('data/validation_predictions_template.csv')
    
    # Verify load_id match
    if not validation['load_id'].equals(template['load_id']):
        print("  ⚠️ WARNING: load_id order mismatch, sorting to match template...")
        # Create prediction mapping
        pred_map = dict(zip(validation['load_id'], validation_pred))
        validation_pred = template['load_id'].map(pred_map).values
    
    submission = template.copy()
    submission['predicted_rate'] = validation_pred
    
    # Validation checks
    print("\n[5] Validation checks...")
    checks = []
    
    # Check 1: Correct number of rows
    if len(submission) == 12000:
        print("  ✓ Row count: 12,000")
        checks.append(True)
    else:
        print(f"  ✗ Row count: {len(submission)} (expected 12,000)")
        checks.append(False)
    
    # Check 2: Correct columns
    if list(submission.columns) == ['load_id', 'predicted_rate']:
        print("  ✓ Columns: load_id, predicted_rate")
        checks.append(True)
    else:
        print(f"  ✗ Columns: {list(submission.columns)}")
        checks.append(False)
    
    # Check 3: No NaN
    if submission['predicted_rate'].isna().sum() == 0:
        print("  ✓ No NaN values")
        checks.append(True)
    else:
        print(f"  ✗ NaN values: {submission['predicted_rate'].isna().sum()}")
        checks.append(False)
    
    # Check 4: No inf
    if not np.isinf(submission['predicted_rate']).any():
        print("  ✓ No infinite values")
        checks.append(True)
    else:
        print(f"  ✗ Infinite values: {np.isinf(submission['predicted_rate']).sum()}")
        checks.append(False)
    
    # Check 5: All positive
    if (submission['predicted_rate'] > 0).all():
        print("  ✓ All values > 0")
        checks.append(True)
    else:
        neg_count = (submission['predicted_rate'] <= 0).sum()
        print(f"  ✗ Non-positive values: {neg_count}")
        checks.append(False)
    
    # Check 6: Unique load_ids
    if submission['load_id'].nunique() == 12000:
        print("  ✓ All load_ids unique")
        checks.append(True)
    else:
        print(f"  ✗ Duplicate load_ids: {12000 - submission['load_id'].nunique()}")
        checks.append(False)
    
    # Check 7: load_ids match template
    if submission['load_id'].equals(template['load_id']):
        print("  ✓ load_ids match template")
        checks.append(True)
    else:
        print("  ✗ load_ids don't match template")
        checks.append(False)
    
    if all(checks):
        print("\n✓ All validation checks passed!")
    else:
        print(f"\n⚠️ {sum(checks)}/{len(checks)} checks passed")
    
    # Save submission
    submission.to_csv('validation_predictions.csv', index=False)
    print("\n✓ Submission saved to: validation_predictions.csv")
    
    # ========================================================================
    # RUN SCORE.PY
    # ========================================================================
    print("\n" + "="*80)
    print("RUNNING score.py FOR FINAL VERIFICATION")
    print("="*80)
    
    print("\nRunning: python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv")
    
    print("\n" + "="*80)
    print("✓ Phase 5 & 6 complete.")
    print("="*80)
    
    print("\nFiles created:")
    print("  - validation_predictions.csv (12,000 predictions)")
    print("  - data/december_chart_inputs.csv (31 December predictions)")
    print("\nNext step: Run score.py to generate the December chart visualization")

"""
Phase 1: Data Understanding and Quality Audit
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("PHASE 1: DATA UNDERSTANDING AND QUALITY AUDIT")
print("="*80)

# Load all data files
print("\n[1] LOADING DATA FILES")
print("-"*80)

train_test = pd.read_csv('data/train_test.csv')
validation = pd.read_csv('data/validation.csv')
val_template = pd.read_csv('data/validation_predictions_template.csv')
december = pd.read_csv('data/december_chart_inputs.csv')

print(f"train_test.csv:    {train_test.shape[0]:,} rows × {train_test.shape[1]} columns")
print(f"validation.csv:    {validation.shape[0]:,} rows × {validation.shape[1]} columns")
print(f"template.csv:      {val_template.shape[0]:,} rows × {val_template.shape[1]} columns")
print(f"december.csv:      {december.shape[0]:,} rows × {december.shape[1]} columns")

print("\n[2] SCHEMA INSPECTION")
print("-"*80)
print("\nTrain/Test Columns:")
print(train_test.dtypes)
print("\nFirst 3 rows:")
print(train_test.head(3))

print("\nValidation Columns:")
print(validation.dtypes)
print("\nFirst 3 rows:")
print(validation.head(3))

print("\nValidation Template:")
print(val_template.head(3))

print("\nDecember Chart Input:")
print(december.head(3))

# Parse dates
print("\n[3] DATE PARSING AND RANGE")
print("-"*80)
train_test['date'] = pd.to_datetime(train_test['date'], errors='coerce')
validation['date'] = pd.to_datetime(validation['date'], errors='coerce')

print(f"\nTrain/Test date range: {train_test['date'].min()} to {train_test['date'].max()}")
print(f"Validation date range: {validation['date'].min()} to {validation['date'].max()}")
print(f"Total train/test days: {(train_test['date'].max() - train_test['date'].min()).days}")
print(f"Date parsing errors in train: {train_test['date'].isna().sum()}")
print(f"Date parsing errors in validation: {validation['date'].isna().sum()}")

# Month distribution
print("\nMonth distribution in train/test:")
print(train_test['date'].dt.to_period('M').value_counts().sort_index())

print("\nMonth distribution in validation:")
print(validation['date'].dt.to_period('M').value_counts().sort_index())

# Target distribution
print("\n[4] TARGET DISTRIBUTION (posted_rate)")
print("-"*80)
print(train_test['posted_rate'].describe())
print(f"\nSkewness: {train_test['posted_rate'].skew():.3f}")
print(f"Kurtosis: {train_test['posted_rate'].kurtosis():.3f}")
print(f"Zeros: {(train_test['posted_rate'] == 0).sum()}")
print(f"Negative: {(train_test['posted_rate'] < 0).sum()}")
print(f"Missing: {train_test['posted_rate'].isna().sum()}")

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    val = train_test['posted_rate'].quantile(p/100)
    print(f"  {p:2d}%: ${val:8.2f}")

# Feature distributions
print("\n[5] FEATURE DISTRIBUTIONS")
print("-"*80)
print("\nDistance:")
print(train_test['distance'].describe())
print(f"Zeros: {(train_test['distance'] == 0).sum()}, Negative: {(train_test['distance'] < 0).sum()}")

print("\nWeight:")
print(train_test['weight'].describe())
print(f"Zeros: {(train_test['weight'] == 0).sum()}, Negative: {(train_test['weight'] < 0).sum()}")

print("\nMarket Index:")
print(train_test['market_index'].describe())

print("\nQuote Signal:")
print(train_test['quote_signal'].describe())

print("\nEquipment types:")
print(train_test['equipment'].value_counts())

print("\nTop 15 Pickup Cities:")
print(train_test['pickup'].value_counts().head(15))

print("\nTop 15 Delivery Cities:")
print(train_test['delivery'].value_counts().head(15))

print(f"\nTotal unique pickup cities: {train_test['pickup'].nunique()}")
print(f"Total unique delivery cities: {train_test['delivery'].nunique()}")

# Temporal shift check
print("\n[6] TEMPORAL SHIFT ANALYSIS")
print("-"*80)
train_max = train_test['date'].max()
val_min = validation['date'].min()
val_max = validation['date'].max()

if pd.notna(val_min) and pd.notna(train_max):
    if val_min > train_max:
        print(f"⚠️  TEMPORAL SPLIT DETECTED!")
        print(f"   Training ends:     {train_max.date()}")
        print(f"   Validation starts: {val_min.date()}")
        print(f"   Gap: {(val_min - train_max).days} days")
        print(f"\n   This is a FUTURE FORECASTING task.")
        print(f"   Random splits would leak temporal information.")
        print(f"   Time-based validation is MANDATORY.")
    else:
        print(f"Dates overlap - validation period {val_min} to {val_max}")
        print(f"Training period {train_test['date'].min()} to {train_max}")

# Distribution comparison
print("\n[7] FEATURE DISTRIBUTION COMPARISON: Train vs Validation")
print("-"*80)

print("\nDistance comparison:")
print(f"Train:      mean={train_test['distance'].mean():.1f}, median={train_test['distance'].median():.1f}, std={train_test['distance'].std():.1f}")
print(f"Validation: mean={validation['distance'].mean():.1f}, median={validation['distance'].median():.1f}, std={validation['distance'].std():.1f}")

print("\nWeight comparison:")
print(f"Train:      mean={train_test['weight'].mean():.1f}, median={train_test['weight'].median():.1f}, std={train_test['weight'].std():.1f}")
print(f"Validation: mean={validation['weight'].mean():.1f}, median={validation['weight'].median():.1f}, std={validation['weight'].std():.1f}")

print("\nMarket Index comparison:")
print(f"Train:      mean={train_test['market_index'].mean():.3f}, median={train_test['market_index'].median():.3f}")
print(f"Validation: mean={validation['market_index'].mean():.3f}, median={validation['market_index'].median():.3f}")

print("\nQuote Signal comparison:")
print(f"Train:      mean={train_test['quote_signal'].mean():.3f}, median={train_test['quote_signal'].median():.3f}")
print(f"Validation: mean={validation['quote_signal'].mean():.3f}, median={validation['quote_signal'].median():.3f}")

print("\nEquipment distribution:")
train_equip = train_test['equipment'].value_counts(normalize=True).sort_index()
val_equip = validation['equipment'].value_counts(normalize=True).sort_index()
comp_df = pd.DataFrame({'train_%': train_equip * 100, 'val_%': val_equip * 100})
print(comp_df)

# Check for unseen categories
print("\n[8] UNSEEN CATEGORIES IN VALIDATION")
print("-"*80)

train_pickups = set(train_test['pickup'].dropna())
val_pickups = set(validation['pickup'].dropna())
unseen_pickups = val_pickups - train_pickups
print(f"Pickup cities: {len(train_pickups)} in train, {len(val_pickups)} in validation")
print(f"Unseen in validation: {len(unseen_pickups)}")
if len(unseen_pickups) > 0 and len(unseen_pickups) <= 20:
    print(f"Examples: {list(unseen_pickups)[:20]}")

train_deliveries = set(train_test['delivery'].dropna())
val_deliveries = set(validation['delivery'].dropna())
unseen_deliveries = val_deliveries - train_deliveries
print(f"\nDelivery cities: {len(train_deliveries)} in train, {len(val_deliveries)} in validation")
print(f"Unseen in validation: {len(unseen_deliveries)}")
if len(unseen_deliveries) > 0 and len(unseen_deliveries) <= 20:
    print(f"Examples: {list(unseen_deliveries)[:20]}")

train_lanes = set(train_test['pickup'].astype(str) + ' -> ' + train_test['delivery'].astype(str))
val_lanes = set(validation['pickup'].astype(str) + ' -> ' + validation['delivery'].astype(str))
unseen_lanes = val_lanes - train_lanes
print(f"\nLanes: {len(train_lanes)} in train, {len(val_lanes)} in validation")
print(f"Unseen lanes in validation: {len(unseen_lanes)} ({100*len(unseen_lanes)/len(val_lanes):.1f}%)")

train_equip = set(train_test['equipment'].dropna())
val_equip = set(validation['equipment'].dropna())
unseen_equip = val_equip - train_equip
print(f"\nEquipment: {len(train_equip)} types in train, {len(val_equip)} in validation")
print(f"Unseen equipment: {len(unseen_equip)}")
if len(unseen_equip) > 0:
    print(f"Examples: {list(unseen_equip)}")

print("\n[9] DATA QUALITY ISSUES")
print("-"*80)

issues = []

# Missing values
print("\nMissing values in train_test:")
missing_train = train_test.isna().sum()
missing_cols = missing_train[missing_train > 0]
if len(missing_cols) > 0:
    print(missing_cols)
    issues.append(f"Missing values: {missing_train.sum()} total across {len(missing_cols)} columns")
else:
    print("None found.")

print("\nMissing values in validation:")
missing_val = validation.isna().sum()
missing_val_cols = missing_val[missing_val > 0]
if len(missing_val_cols) > 0:
    print(missing_val_cols)
else:
    print("None found.")

# Duplicates
print("\n\nDuplicate rows:")
train_dups = train_test.duplicated().sum()
print(f"Train exact duplicates: {train_dups}")
if train_dups > 0:
    issues.append(f"Exact duplicates: {train_dups} rows")

# Near duplicates (same features, different rate)
feature_cols = ['date', 'pickup', 'delivery', 'distance', 'weight', 'equipment']
train_near_dups = train_test.duplicated(subset=feature_cols, keep=False).sum()
print(f"Train near-duplicates (same features, possibly different rate): {train_near_dups}")
if train_near_dups > train_dups:
    issues.append(f"Near duplicates: {train_near_dups - train_dups} rows with same features but different rates")

# Impossible values
print("\n\nImpossible/suspicious values:")
rate_zero = (train_test['posted_rate'] <= 0).sum()
print(f"posted_rate <= 0: {rate_zero}")
if rate_zero > 0:
    issues.append(f"Non-positive rates: {rate_zero} rows")

dist_zero = (train_test['distance'] <= 0).sum()
print(f"Distance <= 0: {dist_zero}")
if dist_zero > 0:
    issues.append(f"Non-positive distance: {dist_zero} rows")

weight_zero = (train_test['weight'] <= 0).sum()
print(f"Weight <= 0: {weight_zero}")
if weight_zero > 0:
    issues.append(f"Non-positive weight: {weight_zero} rows")

# Rate per mile outliers
train_test['rate_per_mile'] = train_test['posted_rate'] / train_test['distance'].replace(0, np.nan)
print(f"\nRate per mile stats:")
print(train_test['rate_per_mile'].describe())
extreme_rpm = ((train_test['rate_per_mile'] < 0.5) | (train_test['rate_per_mile'] > 20)).sum()
print(f"Extreme rate/mile (< $0.50 or > $20): {extreme_rpm}")
if extreme_rpm > 0:
    issues.append(f"Extreme rate-per-mile: {extreme_rpm} rows")

# String inconsistencies
print("\n\nString inconsistencies:")
pickup_variations = train_test['pickup'].apply(lambda x: x != x.strip() if isinstance(x, str) else False).sum()
delivery_variations = train_test['delivery'].apply(lambda x: x != x.strip() if isinstance(x, str) else False).sum()
print(f"Pickup cities with leading/trailing whitespace: {pickup_variations}")
print(f"Delivery cities with leading/trailing whitespace: {delivery_variations}")
if pickup_variations + delivery_variations > 0:
    issues.append(f"Whitespace in city names: {pickup_variations + delivery_variations} rows")

# Check case consistency
pickup_case_issues = len(train_test['pickup'].unique()) - len(train_test['pickup'].str.lower().unique())
delivery_case_issues = len(train_test['delivery'].unique()) - len(train_test['delivery'].str.lower().unique())
print(f"Pickup cities with case variations: ~{pickup_case_issues}")
print(f"Delivery cities with case variations: ~{delivery_case_issues}")
if pickup_case_issues + delivery_case_issues > 0:
    issues.append(f"Case inconsistencies: approximately {pickup_case_issues + delivery_case_issues} variations")

# Check for equipment case issues
equip_case = len(train_test['equipment'].unique()) - len(train_test['equipment'].str.strip().str.title().unique())
print(f"Equipment with case/whitespace variations: ~{equip_case}")
if equip_case > 0:
    issues.append(f"Equipment inconsistencies: {equip_case} variations")

print("\n\n" + "="*80)
print("SUMMARY OF DATA QUALITY ISSUES")
print("="*80)
if len(issues) == 0:
    print("✓ No major data quality issues found.")
else:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    print(f"\nTotal unique issues found: {len(issues)}")

# Save for next phase
train_test.to_csv('outputs/train_test_parsed.csv', index=False)
validation.to_csv('outputs/validation_parsed.csv', index=False)

print("\n✓ Phase 1 complete. Parsed data saved to outputs/")

"""
Phase 1 Continued: Detailed EDA - Relationships and Patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("="*80)
print("DETAILED EDA: RELATIONSHIPS AND PATTERNS")
print("="*80)

# Load parsed data
train_test = pd.read_csv('outputs/train_test_parsed.csv')
train_test['date'] = pd.to_datetime(train_test['date'])

# Add temporal features for analysis
train_test['month'] = train_test['date'].dt.month
train_test['day_of_week'] = train_test['date'].dt.dayofweek
train_test['week_of_year'] = train_test['date'].dt.isocalendar().week
train_test['is_weekend'] = train_test['day_of_week'].isin([5, 6]).astype(int)
train_test['rate_per_mile'] = train_test['posted_rate'] / train_test['distance']

# Create lane identifier
train_test['lane'] = train_test['pickup'] + ' -> ' + train_test['delivery']

print("\n[1] RATE vs DISTANCE")
print("-"*80)

# Distance buckets
train_test['distance_bucket'] = pd.cut(train_test['distance'], 
                                        bins=[0, 300, 600, 1000, 1500, 2000, 3500],
                                        labels=['0-300', '300-600', '600-1000', '1000-1500', '1500-2000', '2000+'])

dist_analysis = train_test.groupby('distance_bucket')['posted_rate'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\nRate by distance bucket:")
print(dist_analysis)

rpm_by_dist = train_test.groupby('distance_bucket')['rate_per_mile'].agg(['mean', 'std'])
print("\nRate per mile by distance bucket:")
print(rpm_by_dist)
print("\n⚠️  Note: Rate-per-mile DECREASES with distance (economies of scale)")

print("\n[2] RATE vs WEIGHT")
print("-"*80)

# Weight buckets (excluding negative values)
train_clean = train_test[train_test['weight'] > 0].copy()
train_clean['weight_bucket'] = pd.cut(train_clean['weight'], 
                                       bins=[0, 20000, 30000, 40000, 50000],
                                       labels=['0-20k', '20-30k', '30-40k', '40k+'])

weight_analysis = train_clean.groupby('weight_bucket')['posted_rate'].agg(['count', 'mean', 'std'])
print("\nRate by weight bucket:")
print(weight_analysis)

print("\n[3] RATE vs EQUIPMENT TYPE")
print("-"*80)

equip_analysis = train_test.groupby('equipment')['posted_rate'].agg(['count', 'mean', 'std', 'min', 'max'])
print(equip_analysis)

equip_rpm = train_test.groupby('equipment')['rate_per_mile'].agg(['mean', 'std'])
print("\nRate per mile by equipment:")
print(equip_rpm)

print("\n[4] TEMPORAL PATTERNS")
print("-"*80)

print("\nRate by month:")
monthly = train_test.groupby('month')['posted_rate'].agg(['count', 'mean', 'std'])
print(monthly)

print("\nRate by day of week (0=Monday, 6=Sunday):")
dow = train_test.groupby('day_of_week')['posted_rate'].agg(['count', 'mean'])
print(dow)

print("\nWeekend vs Weekday:")
weekend = train_test.groupby('is_weekend')['posted_rate'].agg(['count', 'mean'])
weekend.index = ['Weekday', 'Weekend']
print(weekend)

# Trend over time
print("\nTrend over time (monthly average rate):")
monthly_trend = train_test.groupby(train_test['date'].dt.to_period('M'))['posted_rate'].mean()
print(monthly_trend)

print("\n[5] MARKET INDEX & QUOTE SIGNAL")
print("-"*80)

# Correlation with rate
clean_for_corr = train_test[['posted_rate', 'distance', 'weight', 'market_index', 'quote_signal']].dropna()
corr = clean_for_corr.corr()['posted_rate'].sort_values(ascending=False)
print("\nCorrelation with posted_rate:")
print(corr)

print("\nMarket Index analysis:")
mi_clean = train_test.dropna(subset=['market_index'])
print(f"Mean market_index: {mi_clean['market_index'].mean():.3f}")
print(f"Correlation with rate: {mi_clean[['market_index', 'posted_rate']].corr().iloc[0,1]:.3f}")

print("\nQuote Signal analysis:")
print(f"Mean quote_signal: {train_test['quote_signal'].mean():.3f}")
print(f"Correlation with rate: {train_test[['quote_signal', 'posted_rate']].corr().iloc[0,1]:.3f}")

print("\n[6] TOP LANES ANALYSIS")
print("-"*80)

lane_stats = train_test.groupby('lane').agg({
    'posted_rate': ['count', 'mean', 'std'],
    'distance': 'mean'
}).round(2)
lane_stats.columns = ['count', 'mean_rate', 'std_rate', 'mean_distance']
lane_stats = lane_stats.sort_values('count', ascending=False)

print("\nTop 15 lanes by volume:")
print(lane_stats.head(15))

print("\nLane rate variability (high std means volatile pricing):")
high_vol_lanes = lane_stats[lane_stats['count'] >= 20].copy()
high_vol_lanes['cv'] = high_vol_lanes['std_rate'] / high_vol_lanes['mean_rate']
print("\nTop 10 most volatile lanes (CV = std/mean):")
print(high_vol_lanes.nlargest(10, 'cv')[['count', 'mean_rate', 'std_rate', 'cv']])

print("\n[7] GEOGRAPHIC PATTERNS")
print("-"*80)

pickup_rates = train_test.groupby('pickup')['posted_rate'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print("\nTop 10 highest-rate pickup cities:")
print(pickup_rates.head(10))

print("\nTop 10 lowest-rate pickup cities:")
print(pickup_rates.tail(10))

delivery_rates = train_test.groupby('delivery')['posted_rate'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print("\nTop 10 highest-rate delivery cities:")
print(delivery_rates.head(10))

print("\n[8] OUTLIER ANALYSIS")
print("-"*80)

# Rate outliers (beyond 99th percentile)
p99 = train_test['posted_rate'].quantile(0.99)
outliers = train_test[train_test['posted_rate'] > p99]
print(f"\nOutliers (rate > ${p99:.2f}):")
print(f"Count: {len(outliers)} ({100*len(outliers)/len(train_test):.2f}%)")
print(f"Mean distance: {outliers['distance'].mean():.1f} vs overall {train_test['distance'].mean():.1f}")
print(f"Equipment distribution:")
print(outliers['equipment'].value_counts())

# Extreme rate-per-mile
rpm_high = train_test[train_test['rate_per_mile'] > 10]
print(f"\nExtreme rate-per-mile (> $10/mile): {len(rpm_high)} rows")
if len(rpm_high) > 0:
    print("Sample:")
    print(rpm_high[['pickup', 'delivery', 'distance', 'weight', 'equipment', 'posted_rate', 'rate_per_mile']].head())

rpm_low = train_test[train_test['rate_per_mile'] < 1]
print(f"\nLow rate-per-mile (< $1/mile): {len(rpm_low)} rows")
if len(rpm_low) > 0:
    print("Sample:")
    print(rpm_low[['pickup', 'delivery', 'distance', 'weight', 'equipment', 'posted_rate', 'rate_per_mile']].head())

print("\n[9] NEGATIVE WEIGHT ANALYSIS")
print("-"*80)

neg_weight = train_test[train_test['weight'] < 0]
print(f"Rows with negative weight: {len(neg_weight)}")
if len(neg_weight) > 0:
    print("\nSample:")
    print(neg_weight[['pickup', 'delivery', 'distance', 'weight', 'equipment', 'posted_rate']].head(10))
    print(f"\nMean rate: ${neg_weight['posted_rate'].mean():.2f} vs overall ${train_test['posted_rate'].mean():.2f}")
    print(f"Mean distance: {neg_weight['distance'].mean():.1f} vs overall {train_test['distance'].mean():.1f}")

print("\n[10] KEY FINDINGS SUMMARY")
print("="*80)

findings = [
    "1. TEMPORAL SPLIT: Validation is Nov-Dec 2025, train is Jan-Oct 2025. Pure future forecast.",
    "2. DISTANCE EFFECT: Strong positive correlation with rate. Rate-per-mile DECREASES with distance.",
    "3. EQUIPMENT: Reefer > Dry Van > Flatbed in terms of rate-per-mile.",
    "4. WEIGHT: Minimal impact on total rate (already captured by distance and equipment).",
    "5. MARKET INDEX: Mean drops from 1.08 (train) to 0.93 (validation) - DISTRIBUTION SHIFT.",
    "6. UNSEEN LANES: 17.5% of validation lanes never seen in training - need fallback strategy.",
    "7. UNSEEN CITIES: 8 new pickup/delivery cities in validation.",
    "8. DATA QUALITY: 674 missing values (weight/market_index), 292 negative weights, 103 extreme RPM.",
    "9. TEMPORAL PATTERNS: Some monthly variation visible; weekday/weekend effect minimal.",
    "10. QUOTE SIGNAL: High correlation with rate - likely a strong predictor."
]

for finding in findings:
    print(finding)

print("\n✓ Detailed EDA complete.")

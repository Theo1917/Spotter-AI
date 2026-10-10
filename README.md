# Freight Rate Prediction - Spotter ML Assessment

Machine learning solution for predicting freight rates using time-based validation and gradient boosting.

## Project Overview

**Task:** Predict freight rates for 12,000 loads in November-December 2025 based on training data from January-October 2025.

**Key Challenge:** This is a future forecasting task with temporal distribution shifts (14% drop in market index) and 17.5% unseen lanes in the validation set.

**Solution:** Gradient boosting model with time-based validation achieving **$171.52 MAE** (40% improvement over baseline).

## Data

- **Training:** 48,000 loads (Jan-Oct 2025)
- **Validation:** 12,000 loads (Nov-Dec 2025)
- **Features:** pickup/delivery cities, distance, weight, equipment, market_index, quote_signal, lat/lon

## Methodology

### 1. Data Quality
- Fixed 292 negative weights (data entry errors)
- Imputed 674 missing values (weight, market_index)
- Removed 20 extreme outliers from training (rate-per-mile < $0.40 or > $15)

### 2. Validation Strategy ⭐
**Time-based validation is mandatory** because:
- Training data is Jan-Oct 2025, validation is Nov-Dec 2025
- Random splits would leak temporal information (model learns from "the future")
- Real task is forecasting 1-2 months ahead

**Implementation:**
- Primary holdout: Train Jan-Sep (43,129 rows) → Validate Oct (4,851 rows)
- Time-series CV: 3 expanding-window folds
- Comparison: Random K-fold (shows 304 days of temporal overlap - too optimistic)

### 3. Feature Engineering
Created 43 numeric features with **strict leakage protection**:
- **Calendar:** month, day_of_week, holidays, seasonality (12 features)
- **Lane aggregations:** mean/std rate per lane (train-only) with fallbacks for unseen lanes
- **City aggregations:** pickup/delivery statistics
- **Distance/weight:** log transforms, buckets, interactions
- **Equipment:** one-hot indicators

All aggregations computed on training data only, with graceful handling of unseen categories.

### 4. Modeling

Tested 5 models:

| Model | Val MAE | Val R² | Notes |
|-------|---------|--------|-------|
| Baseline (naive RPM) | $287.09 | 0.7807 | Mean rate-per-mile × distance |
| Ridge Regression | $177.55 | 0.8048 | Linear model, all features |
| HistGB (rate) | $178.00 | 0.8016 | Predict rate directly |
| HistGB (log rate) | $174.78 | 0.8044 | Log transform for skewed target |
| **HistGB (RPM)** ⭐ | **$171.52** | **0.8046** | Predict rate-per-mile, multiply by distance |

**Winner:** HistGradientBoosting predicting rate-per-mile
- 40.3% improvement over baseline
- Handles missing values natively
- Robust to unseen categories
- Shift-aware predictions (mean $2,438 vs training $2,374)

## Key Findings

1. **Distance is king:** 0.91 correlation with rate, but rate-per-mile *decreases* with distance (economies of scale)
2. **Lane matters:** Lane history is highly predictive, but 17.5% of validation lanes are unseen
3. **Market shift:** Market index drops 14% in validation period (distribution shift handled via feature)
4. **Equipment hierarchy:** Reefer ($2.38/mi) > Flatbed ($2.29/mi) > Dry Van ($2.12/mi)
5. **Temporal patterns:** Minimal day-of-week effect; monthly seasonality present but weak

## Results

### Validation Predictions (validation_predictions.csv)
- 12,000 predictions for Nov-Dec 2025
- Mean: $2,438.31 (vs training $2,374)
- All quality checks passed ✓

### December Chart (data/december_chart_inputs.csv)
- 31 predictions for fixed lane (Lexington → Fort Wayne, 360 mi, Dry Van)
- Predictions: $819.03 ± $0.27
- Note: Flat predictions expected for fixed lane (only date varies)

## Limitations & Future Work

1. **Flat December chart:** Model learned weak calendar effects. Could improve with:
   - Stronger holiday features (days-until-holiday)
   - Separate time series component
   - Ensemble with seasonal models

2. **Distribution shift:** 14% market index drop in validation. Model uses it as a feature but could:
   - Use market index to scale predictions
   - Train separate models by market regime
   - Monitor drift and retrain more frequently

3. **Segment performance:** Long-haul and Reefer loads have higher errors. Could improve with:
   - Segment-specific models
   - Separate volatility predictions
   - More lane-level history

4. **Feature engineering:** Could add:
   - Region/state groupings
   - Fuel price indicators
   - Weather data
   - Competitive dynamics (e.g., broker density)

# Loom Video Script (2-3 minutes)
**Freight Rate Prediction - Spotter ML Assessment**

---

## Introduction (15 seconds)
"Hi, I'm presenting my solution for predicting freight rates for Spotter. I built a gradient boosting model that achieves a validation MAE of $171, which is a 40% improvement over the baseline. Let me walk you through the key aspects."

---

## 1. Key EDA Findings (30 seconds)

**[Show: outputs/phase1_summary.txt or notebooks/02_detailed_eda.py output]**

"First, the data analysis revealed five critical insights:

**One** - This is a temporal forecasting task. Training data is January through October 2025, and validation is November-December. There's a clean 1-day gap between them.

**Two** - Distance is the king predictor with a 0.91 correlation to rate, but interestingly, the rate-per-mile *decreases* as distance increases - from $2.74 per mile for short hauls down to $1.94 for long hauls. This is the classic economies of scale in trucking.

**Three** - The market index drops 14% in the validation period, signaling a softer freight market.

**Four** - 17.5% of validation lanes never appeared in training, plus 8 completely new cities.

**Five** - Equipment matters: Reefer loads cost $2.38 per mile versus Dry Van at $2.12."

---

## 2. Data Quality Issues & Fixes (25 seconds)

**[Show: src/data_cleaning.py]**

"For data quality, I found and fixed three main issues:

**First** - 292 rows had negative weights, which is physically impossible. These were clearly data entry errors with wrong signs, so I took the absolute value.

**Second** - 674 missing values across weight and market index columns. I imputed these with the median to preserve the distribution.

**Third** - 103 extreme outliers with rate-per-mile below 40 cents or above $15. I removed these 20 rows from training as they were likely errors, but kept all validation rows and corrected them instead of dropping."

---

## 3. Model Selection & Training Approach (40 seconds)

**[Show: outputs/model_comparison.csv]**

"I tested five models, and here's why I chose gradient boosting predicting rate-per-mile:

**The models** - I compared a naive baseline, Ridge regression, and three variants of HistGradientBoosting: predicting rate directly, log of rate, and rate-per-mile.

**The winner** - Rate-per-mile approach performed best at $171.52 MAE. Why rate-per-mile? It normalizes by distance, which reduces target variance and lets the model better capture that non-linear relationship where cost-per-mile decreases with distance.

**Why gradient boosting over linear models?** Three reasons: It captures non-linearities like that decreasing rate-per-mile curve. It handles missing values natively without extra preprocessing. And it's robust to outliers. Ridge regression got $177 MAE versus our $171, so the non-linearity matters."

---

## 4. Validation Strategy (CRITICAL - 40 seconds)

**[Show: outputs/phase2_summary.txt or src/validation_strategy.py output]**

"The validation strategy is the most critical part of this project, and I want to emphasize why:

**The problem** - This is pure future forecasting. The validation data comes *after* the training data chronologically. If I used random K-fold cross-validation, the model would train on data from September and October, then validate on data from January and February in the same fold. That creates 304 days of temporal overlap.

**What that means** - The model would learn from future loads in the same lanes and time periods. It would memorize patterns that wouldn't be available in production. Metrics would be artificially optimistic.

**My approach** - Time-based validation. I trained on January through September and validated on October. This mimics production: train on the past, predict the future. No temporal leakage. I also ran 3-fold expanding-window time-series cross-validation for robustness.

**The proof** - I ran random K-fold just to demonstrate the problem. It showed all 5 folds had complete temporal overlap, which would give misleading results.

This validation strategy is heavily graded because it's the difference between a model that looks good on paper versus one that actually works in production."

---

## 5. Code Walkthrough (30 seconds)

**[Show: file structure and main scripts]**

"The code is organized into five modular scripts:

**data_cleaning.py** handles the 674 missing values and 292 negative weights we discussed.

**feature_engineering.py** creates 43 features with strict leakage protection - all lane and city statistics are computed on training data only, with a 3-tier fallback for unseen categories.

**validation_strategy.py** implements the time-based split and generates both the primary holdout and time-series CV folds.

**modeling.py** trains and compares all five models, selects the best one, and saves it.

**predictions.py** loads the trained model, generates the 12,000 validation predictions and 31 December chart predictions.

Everything is reproducible with fixed random seeds. You can run the entire pipeline with one command: python src/run_pipeline.py."

---

## Closing (10 seconds)

"The final model achieves $171 MAE with proper time-based validation, handles 17.5% unseen lanes gracefully, and produces well-calibrated predictions. All deliverables are ready for submission. Thank you for your time."

---

## TIMING BREAKDOWN
- Introduction: 15s
- EDA Findings: 30s  
- Data Quality: 25s
- Model Selection: 40s
- Validation Strategy: 40s
- Code Walkthrough: 30s
- Closing: 10s
**Total: 190 seconds = 3 minutes 10 seconds** ✓

---

## SCREEN SHARING GUIDE

### Segment 1: EDA Findings (30s)
- **Show:** outputs/phase1_summary.txt (scroll to "KEY RELATIONSHIPS" section)
- **OR:** Terminal output from notebooks/02_detailed_eda.py
- **Highlight:** The temporal split dates, distance correlation, rate-per-mile by distance bucket

### Segment 2: Data Quality (25s)
- **Show:** src/data_cleaning.py (open in editor)
- **Focus on:** Lines where negative weights are fixed (abs value), missing value imputation, outlier removal
- **Briefly show:** The cleaning summary output

### Segment 3: Model Selection (40s)
- **Show:** outputs/model_comparison.csv (in Excel/spreadsheet or terminal)
- **Highlight:** The 5 rows showing all models, MAE column, best model highlighted
- **Briefly show:** outputs/segment_analysis.csv for performance by segment

### Segment 4: Validation Strategy (40s)
- **Show:** outputs/phase2_summary.txt (open in text editor)
- **Scroll to:** "WHY TIME-BASED VALIDATION IS MANDATORY" section
- **Also show:** outputs/cv_folds_summary.csv and random_folds_summary.csv side by side
- **Highlight:** Zero temporal overlap in time-based vs 304 days in random

### Segment 5: Code Walkthrough (30s)
- **Show:** Project folder structure (file explorer or `tree` command)
- **Quickly open:** Each of the 5 main scripts (don't read, just show they exist)
- **Focus on:** src/ folder structure, clean organization
- **End with:** README.md showing the one-command pipeline run

### Closing (10s)
- **Show:** validation_predictions.csv (first few rows)
- **Show:** outputs/model_info.json (final metrics)

---

## KEY TALKING POINTS TO EMPHASIZE

1. **Temporal validation** - This is THE most important decision. Hammer home why random splits are wrong.
2. **Practical results** - $171 MAE, 40% improvement, well-calibrated predictions
3. **Robust to unseen data** - 17.5% unseen lanes handled with zero NaN values
4. **Clean code** - Modular, reproducible, one-command pipeline
5. **Domain understanding** - Rate-per-mile decreases with distance (economies of scale)

---

## CONFIDENCE BOOSTERS

If you need to pause or gather thoughts:
- "Let me show you exactly how this works..." (buys time)
- "This is a critical point..." (emphasizes importance)
- "The key insight here is..." (refocuses attention)

Remember: You know this material cold. You've solved a real forecasting problem correctly. Be confident!

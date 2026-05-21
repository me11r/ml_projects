# Endterm Project: Data Curation Analysis & Model Optimization

## Executive Summary

Following professor's recommendation to analyze dataset and identify patterns in well-predicted samples, we conducted a comprehensive data curation study. Key findings:

- **Good predictions (R² = 0.08-0.12) are concentrated on lower CLTV values** (mean: 63k vs 104k for bad predictions)
- **Filtering or weighting data by "good predictions" REDUCES overall model performance**
- **Baseline model is optimal** - needs full spectrum of data for generalization
- **Expected R² ceiling: ~0.12-0.15** with current features and data quality

---

## Part 1: Initial Analysis - Identifying "Good" Predictions

### Step 1.1: What Makes a Prediction "Good"?

We trained a baseline model and analyzed residuals to identify the bottom 15% (best predictions):

```
Train Set:
  Good samples: 9,386 (15%)
  Mean residual: 2,163 (vs 37,040 for bad predictions)
  
Test Set:
  Good samples: 4,023 (15%)
  Mean residual: 3,781 (vs 53,495 for bad predictions)
```

### Step 1.2: Feature Patterns in "Good" Predictions

| Feature | Good % | Bad % | Insight |
|---------|--------|-------|---------|
| **Single Policy** | 56% | 28% | 2x more likely good |
| **Rural Area** | 37% | 29% | Slightly more predictable |
| **Platinum Policy** | 48% | 54% | Lower-value policies better |
| **CLTV <= 50k** | 42% | 20% | 2.17x ratio - STRONG PATTERN |
| **CLTV 50-100k** | 28% | 43% | Prediction gets harder |

### Step 1.3: The Key Insight - CLTV Distribution

**Good predictions have MUCH LOWER CLTV values:**

```
Good Predictions:
  Mean: 63,712
  Median: 55,458
  Range: [28k, 201k]

Bad Predictions:
  Mean: 104,346
  Median: 67,032
  Range: [25k, 724k]
```

The model is simply better at predicting **lower, more stable CLTV values**.

---

## Part 2: Testing Data Curation Strategies

### Strategy 1: Filter Training Data to "Good Predictions Only"
- Dataset size: 89k → 9.4k samples
- **Result: R² DECREASED from 0.111 → 0.097** 
- Why? Overfitting to subset, loses patterns for high-CLTV customers

### Strategy 2: Mixed Dataset (70% good + 30% bad)
- Dataset size: 89k → 13.4k samples
- **Result: R² DECREASED from 0.111 → 0.099** 
- Still too little data from extreme values

### Strategy 3: Feature Hint ("is_predictable" binary feature)
- Added indicator: 1 if sample was "good prediction", 0 otherwise
- **Result: R² INCREASED from 0.115 → 0.117** ✓ (+1.7%)
- Modest but measurable improvement

### Strategy 4: Low CLTV Range (CLTV ≤ 100k)
- Dataset size: 89k → 19.5k samples
- **Result: R² DECREASED from 0.111 → 0.097** 
- Model needs outliers for balance

### Strategy 5: Very Low CLTV Range (CLTV ≤ 50k)
- Dataset size: 89k → 12.4k samples
- **Result: R² COLLAPSED to -0.48** 
- Severe overfitting, predictions are meaningless

### Strategy 6: Weighted Training (Up-weight good predictions)
- Weights: Good = 2x, Bad = 1x
- **Result: R² stayed ~0.113** (no improvement) 
- Good predictions are only 15%, hard to over-weight significantly

### Strategy 7: Weighted Training (Aggressive)
- Weights: Good = 3x, Bad = 0.5x
- **Result: R² DECREASED to 0.106** 
- Makes model focus too much on minority class

---

## Part 3: Optimal Solution

### Best Approach: Keep Full Dataset with Optimized Hyperparameters

```python
GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=6,
    min_samples_leaf=3,
    subsample=0.85,
    random_state=42
)
```

**Performance:**
- R² = 0.1152
- RMSE = 84,384
- MAE = 44,402

### Ensemble Approach (Weighted average)

Combining optimized GB and RF:
- R² = 0.1132
- Slightly lower than pure GB, but more robust

---

## Part 4: Why Data Curation Failed

### Root Cause 1: High Variance in Target Variable
- CLTV ranges from 25k to 720k (29x range!)
- Some values are inherently unpredictable
- Model needs to "see" extremes to learn proper scaling

### Root Cause 2: Good Predictions Are Cluster, Not Pattern
- "Good" predictions happen to cluster in low-CLTV region
- This is correlation, not causation
- Filtering by correlation → overfitting

### Root Cause 3: Limited Features
- Only 10 original features
- Missing information about customer lifetime value drivers
- With such sparse data, every sample matters

---

## Part 5: Analysis by CLTV Segments

Using best model (Baseline GB with R²=0.1152):

| CLTV Range | Samples | R² Score | MAE | Quality |
|-----------|---------|----------|-----|---------|
| 0-50k | 6,225 | -17.64 | 17,251 | Poor fit |
| 50-100k | 13,155 | -4.60 | 24,566 | Poor fit |
| 100-200k | 4,888 | -1.36 | 34,274 | Poor fit |
| 200k+ | 2,550 | -5.39 | 234,555 | Very Poor |
| **Overall** | **26,818** | **0.1152** | **44,402** | Moderate |

**Note:** Negative R² in segments means predictions are worse than simply predicting the mean. This indicates **poor local fit** in each segment.

---

## Part 6: Recommendations for Further Improvement

### Short Term (Quick Wins)
1. ✓ Use log transformation of target (already done)
2. ✓ Hyperparameter tuning (optimized GB parameters)
3. ✓ Feature scaling with StandardScaler (already done)
4. Consider: Feature hint approach (+1-2%)

### Medium Term (Better Models)
1. **Try XGBoost or LightGBM** - Often outperform GB by 5-10%
2. **Create interaction features** - `num_policies × claim_amount`, etc.
3. **Segment-specific models** - Train separate models for CLTV ranges
4. **Neural networks** - Might capture complex patterns

### Long Term (Fundamental Improvements)
1. **Collect more features** - Current 10 features may be insufficient
2. **Temporal data** - Customer history, trends
3. **Deeper domain analysis** - Business rules for value prediction
4. **Customer segmentation** - Build different models per segment

---

## Part 7: Key Learnings About Data Curation

### What WORKS ✓
- Adding informative features ("is_predictable" +1.7%)
- Hyperparameter tuning
- Ensemble methods
- Using full dataset for generalization

### What DOESN'T WORK 
- Filtering to "good" samples (overfitting)
- Over-weighting minorities (still too small a signal)
- Training on subsets of data (loss of information)
- Focusing only on one segment

### General Principle
**Data curation works best when:**
- You're removing TRUE NOISE or errors
- You have 10x+ more data than you need
- The "good" samples represent a clear, generalizable pattern

**Data curation fails when:**
- "Good" is relative, not absolute
- The pattern is local, not global
- You have limited data to begin with

---

## Conclusion

The professor's suggestion to analyze "good" predictions was valuable for understanding the model's behavior. We found that:

1. The model naturally performs better on lower-CLTV customers
2. Attempting to force this pattern through data filtering HURTS performance
3. The optimal solution is to **keep all data and optimize hyperparameters**
4. The current ceiling is R² ≈ 0.12, suggesting data/feature limitations

**Final Model: Baseline GradientBoosting**
- Trained on full 62.6k samples
- R² = 0.1152
- This is the best we can achieve with current approach

For significant improvements (R² > 0.25), we need:
- Better features (domain knowledge)
- More data (temporal, behavioral)
- Alternative modeling approaches (neural networks, segment-specific models)


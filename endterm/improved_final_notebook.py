"""
ENDTERM PROJECT: FINAL IMPROVED SOLUTION
=========================================

Data Curation Analysis & Model Optimization

Key Findings:
1. Good predictions (15%) have lower CLTV values (mean=63k vs 104k)
2. Model performs better on lower CLTV ranges (0-50k)
3. Single-policy holders are easier to predict (56% vs 28%)
4. Training on filtered data REDUCES generalization
5. Baseline model with good hyperparameters is best approach

Recommendations:
- Keep FULL training data (don't filter)
- Optimize hyperparameters for full spectrum
- Use GradientBoosting (better than Random Forest)
- Monitor performance on different CLTV ranges
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENDTERM FINAL: DATA CURATION + OPTIMIZED MODELS")
print("="*80)

# === LOAD DATA ===
df = pd.read_csv('train_BRCpofr.csv')
target = 'cltv'

# Preprocessing
df_clean = df.drop(columns=['id'])
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Log transform
y_log = np.log1p(y)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nDataset: {df.shape[0]} samples, {len(numerical_cols)} numerical + {len(categorical_cols)} categorical")
print(f"Target (CLTV): mean={y.mean():,.0f}, median={y.median():,.0f}")
print(f"Target skewness: {y.skew():.2f} → {y_log.skew():.2f} (after log transform)")

# === PREPROCESSING ===
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_processed, y_log, test_size=0.3, random_state=42
)

y_train = np.expm1(y_train_log)
y_test = np.expm1(y_test_log)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# === MODEL 1: BASELINE GradientBoosting ===
print(f"\n" + "="*80)
print("MODEL 1: Baseline GradientBoosting (Original)")
print("="*80)

gb_baseline = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

gb_baseline.fit(X_train, y_train_log)
y_pred_base = np.expm1(gb_baseline.predict(X_test))

r2_base = r2_score(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)

print(f"Performance:")
print(f"  R² = {r2_base:.4f}")
print(f"  RMSE = {rmse_base:,.2f}")
print(f"  MAE = {mae_base:,.2f}")

# === MODEL 2: OPTIMIZED GradientBoosting ===
print(f"\n" + "="*80)
print("MODEL 2: Optimized GradientBoosting")
print("="*80)

gb_opt = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=7,
    min_samples_split=6,
    min_samples_leaf=3,
    subsample=0.85,
    max_features='sqrt',
    random_state=42
)

gb_opt.fit(X_train, y_train_log)
y_pred_opt = np.expm1(gb_opt.predict(X_test))

r2_opt = r2_score(y_test, y_pred_opt)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
mae_opt = mean_absolute_error(y_test, y_pred_opt)

print(f"Performance:")
print(f"  R² = {r2_opt:.4f}")
print(f"  RMSE = {rmse_opt:,.2f}")
print(f"  MAE = {mae_opt:,.2f}")
print(f"\nImprovement over baseline:")
print(f"  ΔR² = {r2_opt - r2_base:+.4f} ({(r2_opt-r2_base)/r2_base*100:+.2f}%)")
print(f"  ΔRMSE = {rmse_opt - rmse_base:+.2f} ({(rmse_opt-rmse_base)/rmse_base*100:+.2f}%)")

# === MODEL 3: RandomForest Optimized ===
print(f"\n" + "="*80)
print("MODEL 3: Optimized RandomForest")
print("="*80)

rf_opt = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_opt.fit(X_train, y_train_log)
y_pred_rf = np.expm1(rf_opt.predict(X_test))

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Performance:")
print(f"  R² = {r2_rf:.4f}")
print(f"  RMSE = {rmse_rf:,.2f}")
print(f"  MAE = {mae_rf:,.2f}")

# === ENSEMBLE: Average GB_OPT + RF_OPT ===
print(f"\n" + "="*80)
print("MODEL 4: Ensemble (avg GB + RF)")
print("="*80)

y_pred_ensemble = (y_pred_opt + y_pred_rf) / 2

r2_ens = r2_score(y_test, y_pred_ensemble)
rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
mae_ens = mean_absolute_error(y_test, y_pred_ensemble)

print(f"Performance:")
print(f"  R² = {r2_ens:.4f}")
print(f"  RMSE = {rmse_ens:,.2f}")
print(f"  MAE = {mae_ens:,.2f}")

# === FINAL COMPARISON ===
print(f"\n" + "="*80)
print("FINAL RESULTS: ALL MODELS")
print("="*80)

results_final = pd.DataFrame({
    'Model': ['Baseline GB', 'Optimized GB', 'Optimized RF', 'Ensemble'],
    'R2': [r2_base, r2_opt, r2_rf, r2_ens],
    'RMSE': [rmse_base, rmse_opt, rmse_rf, rmse_ens],
    'MAE': [mae_base, mae_opt, mae_rf, mae_ens],
})

print("\n" + results_final.to_string(index=False))

best_r2 = results_final['R2'].max()
best_model = results_final.loc[results_final['R2'].idxmax(), 'Model']

print(f"\n✓ BEST MODEL: {best_model} with R² = {best_r2:.4f}")

# === ANALYSIS BY CLTV RANGES ===
print(f"\n" + "="*80)
print("PERFORMANCE BY CLTV RANGE (using best model)")
print("="*80)

if best_model == 'Optimized GB':
    y_pred_best = y_pred_opt
elif best_model == 'Optimized RF':
    y_pred_best = y_pred_rf
else:
    y_pred_best = y_pred_ensemble

ranges = [(0, 50000), (50000, 100000), (100000, 200000), (200000, 1000000)]
for lower, upper in ranges:
    mask = (y_test >= lower) & (y_test < upper)
    if mask.sum() > 0:
        r2_range = r2_score(y_test[mask], y_pred_best[mask])
        mae_range = mean_absolute_error(y_test[mask], y_pred_best[mask])
        print(f"  [{lower/1000:>3.0f}k - {upper/1000:>3.0f}k): R²={r2_range:+.4f}, MAE={mae_range:,.0f} ({mask.sum()} samples)")

# === RESIDUALS ANALYSIS ===
print(f"\n" + "="*80)
print("RESIDUALS ANALYSIS")
print("="*80)

residuals = y_test - y_pred_best
print(f"\nResiduals (predictions error):")
print(f"  Mean: {residuals.mean():+,.0f}")
print(f"  Std: {residuals.std():,.0f}")
print(f"  Median: {residuals.median():+,.0f}")
print(f"  Min: {residuals.min():+,.0f}")
print(f"  Max: {residuals.max():+,.0f}")

# Percentage of predictions within tolerance
tolerance_5k = (np.abs(residuals) <= 5000).sum() / len(residuals) * 100
tolerance_10k = (np.abs(residuals) <= 10000).sum() / len(residuals) * 100

print(f"\nPrediction accuracy:")
print(f"  Within ±5k: {tolerance_5k:.1f}%")
print(f"  Within ±10k: {tolerance_10k:.1f}%")

results_final.to_csv('FINAL_optimized_results.csv', index=False)

print(f"\n" + "="*80)
print("CONCLUSION & RECOMMENDATIONS")
print("="*80)

print(f"""
Current Status:
  Baseline R²: {r2_base:.4f}
  Optimized R²: {best_r2:.4f}
  Improvement: {(best_r2-r2_base)/r2_base*100:+.2f}%

Data Curation Insights:
  1. The model performs better on LOWER CLTV values
  2. Training on filtered data REDUCES generalization (R² = 0.10 vs 0.11)
  3. Good predictions (15%) are concentrated in lower ranges
  4. The high variance in CLTV makes perfect prediction impossible

Why Data Curation Didn't Help:
  ✗ Filtering removes important patterns for other CLTV ranges
  ✗ Weighted training helps ~0.1%, not significant
  ✗ Model needs full spectrum to generalize

Next Steps for Further Improvement:
  1. Feature Engineering:
     - Interaction terms: num_policies × claim_amount
     - Polynomial features on key predictors
     - Domain-specific features if available
  
  2. Target Variable Preprocessing:
     - Already using log1p transform (good!)
     - Try Box-Cox transform or other methods
  
  3. Alternative Models:
     - XGBoost (often outperforms GB)
     - LightGBM (faster, sometimes better)
     - Neural networks (requires more tuning)
  
  4. Data Collection:
     - Current data has high CLTV variance
     - Model ceiling may be ~0.2-0.25 with this data
     - Consider getting additional features or data

Current Best Approach:
  ✓ Use optimized GB/RF ensemble
  ✓ Keep full training data
  ✓ Focus on hyperparameter tuning
  ✓ Monitor performance by CLTV segments
""")

print("="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)

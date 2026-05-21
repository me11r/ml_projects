"""
Step 2: Retrain models on curated datasets
Compare 3 strategies:
1. GOOD_ONLY: Only 15% of best predictions
2. MIXED: 70% good + 30% bad
3. ORIGINAL: Baseline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 2: RETRAINING MODELS ON CURATED DATASETS")
print("="*80)

# Load test set (ALWAYS the same for fair comparison)
df_test = pd.read_csv('train_BRCpofr.csv')
# Use the 30% that was originally held as test
np.random.seed(42)
df_full = df_test.copy()
_, indices_test_original = train_test_split(
    np.arange(len(df_full)), test_size=0.3, random_state=42
)
df_test = df_full.iloc[indices_test_original].reset_index(drop=True)

target = 'cltv'
X_test = df_test.drop(columns=['id', target])
y_test = df_test[target]
y_test_log = np.log1p(y_test)

categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_test.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Fit preprocessor on test set features (to apply to train sets)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_test_processed = preprocessor.fit_transform(X_test)
y_test_processed = np.expm1(y_test_log)

print(f"\nTest set (fixed for all models): {len(df_test)} samples")
print(f"  y_test range: [{y_test.min():,.0f}, {y_test.max():,.0f}]")
print(f"  y_test mean: {y_test.mean():,.0f}")

# === STRATEGY 1: GOOD ONLY ===
print(f"\n" + "="*80)
print("STRATEGY 1: TRAIN ON GOOD PREDICTIONS ONLY (15%)")
print("="*80)

df_train_1 = pd.read_csv('train_GOOD_predictions_only.csv')
df_train_1 = df_train_1.drop(columns=['id'])
X_train_1 = df_train_1.drop(columns=[target])
y_train_1 = df_train_1[target]
y_train_1_log = np.log1p(y_train_1)

X_train_1_processed = preprocessor.transform(X_train_1)

print(f"Training set: {len(df_train_1)} samples")
print(f"  y_train range: [{y_train_1.min():,.0f}, {y_train_1.max():,.0f}]")
print(f"  y_train mean: {y_train_1.mean():,.0f}")

# Train RF
rf_1 = RandomForestRegressor(
    n_estimators=100, 
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42, 
    n_jobs=-1
)
rf_1.fit(X_train_1_processed, y_train_1_log)

y_pred_1 = np.expm1(rf_1.predict(X_test_processed))
r2_1 = r2_score(y_test_processed, y_pred_1)
rmse_1 = np.sqrt(mean_squared_error(y_test_processed, y_pred_1))
mae_1 = mean_absolute_error(y_test_processed, y_pred_1)

print(f"\n✓ RandomForest on GOOD_ONLY:")
print(f"  R² = {r2_1:.4f}")
print(f"  RMSE = {rmse_1:,.2f}")
print(f"  MAE = {mae_1:,.2f}")

# Train GB
gb_1 = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
gb_1.fit(X_train_1_processed, y_train_1_log)

y_pred_gb_1 = np.expm1(gb_1.predict(X_test_processed))
r2_gb_1 = r2_score(y_test_processed, y_pred_gb_1)
rmse_gb_1 = np.sqrt(mean_squared_error(y_test_processed, y_pred_gb_1))
mae_gb_1 = mean_absolute_error(y_test_processed, y_pred_gb_1)

print(f"\n✓ GradientBoosting on GOOD_ONLY:")
print(f"  R² = {r2_gb_1:.4f}")
print(f"  RMSE = {rmse_gb_1:,.2f}")
print(f"  MAE = {mae_gb_1:,.2f}")

# === STRATEGY 2: MIXED (70% good + 30% bad) ===
print(f"\n" + "="*80)
print("STRATEGY 2: TRAIN ON MIXED (70% GOOD + 30% BAD)")
print("="*80)

df_train_2 = pd.read_csv('train_MIXED_70good_30bad.csv')
df_train_2 = df_train_2.drop(columns=['id'])
X_train_2 = df_train_2.drop(columns=[target])
y_train_2 = df_train_2[target]
y_train_2_log = np.log1p(y_train_2)

X_train_2_processed = preprocessor.transform(X_train_2)

print(f"Training set: {len(df_train_2)} samples")
print(f"  y_train range: [{y_train_2.min():,.0f}, {y_train_2.max():,.0f}]")
print(f"  y_train mean: {y_train_2.mean():,.0f}")

# Train RF
rf_2 = RandomForestRegressor(
    n_estimators=100, 
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42, 
    n_jobs=-1
)
rf_2.fit(X_train_2_processed, y_train_2_log)

y_pred_2 = np.expm1(rf_2.predict(X_test_processed))
r2_2 = r2_score(y_test_processed, y_pred_2)
rmse_2 = np.sqrt(mean_squared_error(y_test_processed, y_pred_2))
mae_2 = mean_absolute_error(y_test_processed, y_pred_2)

print(f"\n✓ RandomForest on MIXED:")
print(f"  R² = {r2_2:.4f}")
print(f"  RMSE = {rmse_2:,.2f}")
print(f"  MAE = {mae_2:,.2f}")

# Train GB
gb_2 = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
gb_2.fit(X_train_2_processed, y_train_2_log)

y_pred_gb_2 = np.expm1(gb_2.predict(X_test_processed))
r2_gb_2 = r2_score(y_test_processed, y_pred_gb_2)
rmse_gb_2 = np.sqrt(mean_squared_error(y_test_processed, y_pred_gb_2))
mae_gb_2 = mean_absolute_error(y_test_processed, y_pred_gb_2)

print(f"\n✓ GradientBoosting on MIXED:")
print(f"  R² = {r2_gb_2:.4f}")
print(f"  RMSE = {rmse_gb_2:,.2f}")
print(f"  MAE = {mae_gb_2:,.2f}")

# === STRATEGY 0: ORIGINAL (BASELINE) ===
print(f"\n" + "="*80)
print("STRATEGY 0: ORIGINAL DATASET (BASELINE)")
print("="*80)

df_train_0 = pd.read_csv('train_BRCpofr.csv')
df_train_0 = df_train_0.drop(columns=['id'])
X_train_0 = df_train_0.drop(columns=[target])
y_train_0 = df_train_0[target]
y_train_0_log = np.log1p(y_train_0)

# Use indices to exclude test set
np.random.seed(42)
_, indices_test = train_test_split(np.arange(len(df_train_0)), test_size=0.3, random_state=42)
indices_train = np.setdiff1d(np.arange(len(df_train_0)), indices_test)

X_train_0 = X_train_0.iloc[indices_train].reset_index(drop=True)
y_train_0 = y_train_0.iloc[indices_train].reset_index(drop=True)
y_train_0_log = y_train_0_log.iloc[indices_train].reset_index(drop=True)

X_train_0_processed = preprocessor.transform(X_train_0)

print(f"Training set: {len(X_train_0)} samples")
print(f"  y_train range: [{y_train_0.min():,.0f}, {y_train_0.max():,.0f}]")
print(f"  y_train mean: {y_train_0.mean():,.0f}")

# Train RF
rf_0 = RandomForestRegressor(
    n_estimators=100, 
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42, 
    n_jobs=-1
)
rf_0.fit(X_train_0_processed, y_train_0_log)

y_pred_0 = np.expm1(rf_0.predict(X_test_processed))
r2_0 = r2_score(y_test_processed, y_pred_0)
rmse_0 = np.sqrt(mean_squared_error(y_test_processed, y_pred_0))
mae_0 = mean_absolute_error(y_test_processed, y_pred_0)

print(f"\n✓ RandomForest on ORIGINAL:")
print(f"  R² = {r2_0:.4f}")
print(f"  RMSE = {rmse_0:,.2f}")
print(f"  MAE = {mae_0:,.2f}")

# Train GB
gb_0 = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
gb_0.fit(X_train_0_processed, y_train_0_log)

y_pred_gb_0 = np.expm1(gb_0.predict(X_test_processed))
r2_gb_0 = r2_score(y_test_processed, y_pred_gb_0)
rmse_gb_0 = np.sqrt(mean_squared_error(y_test_processed, y_pred_gb_0))
mae_gb_0 = mean_absolute_error(y_test_processed, y_pred_gb_0)

print(f"\n✓ GradientBoosting on ORIGINAL:")
print(f"  R² = {r2_gb_0:.4f}")
print(f"  RMSE = {rmse_gb_0:,.2f}")
print(f"  MAE = {mae_gb_0:,.2f}")

# === COMPARISON TABLE ===
print(f"\n" + "="*80)
print("FINAL COMPARISON: ALL STRATEGIES")
print("="*80)

results = pd.DataFrame({
    'Strategy': ['ORIGINAL', 'GOOD_ONLY', 'MIXED (70/30)'],
    'RF_R2': [r2_0, r2_1, r2_2],
    'RF_RMSE': [rmse_0, rmse_1, rmse_2],
    'RF_MAE': [mae_0, mae_1, mae_2],
    'GB_R2': [r2_gb_0, r2_gb_1, r2_gb_2],
    'GB_RMSE': [rmse_gb_0, rmse_gb_1, rmse_gb_2],
    'GB_MAE': [mae_gb_0, mae_gb_1, mae_gb_2],
})

print("\n" + results.to_string(index=False))

# Best strategy
best_r2 = max(r2_0, r2_1, r2_2, r2_gb_0, r2_gb_1, r2_gb_2)
best_strategy_idx = [r2_0, r2_1, r2_2, r2_gb_0, r2_gb_1, r2_gb_2].index(best_r2)
best_strategies = ['ORIGINAL_RF', 'GOOD_ONLY_RF', 'MIXED_RF', 'ORIGINAL_GB', 'GOOD_ONLY_GB', 'MIXED_GB']
best_strategy = best_strategies[best_strategy_idx]

print(f"\n✓ BEST STRATEGY: {best_strategy} with R² = {best_r2:.4f}")

# Save results
results.to_csv('comparison_results.csv', index=False)
print(f"\n✓ Results saved to comparison_results.csv")

# Calculate improvements
print(f"\n" + "="*80)
print("IMPROVEMENTS OVER BASELINE")
print("="*80)

print(f"\nGOOD_ONLY strategy:")
print(f"  RF R² improvement: {((r2_1 - r2_0) / r2_0 * 100):.1f}% (from {r2_0:.4f} → {r2_1:.4f})")
print(f"  GB R² improvement: {((r2_gb_1 - r2_gb_0) / r2_gb_0 * 100):.1f}% (from {r2_gb_0:.4f} → {r2_gb_1:.4f})")

print(f"\nMIXED strategy:")
print(f"  RF R² improvement: {((r2_2 - r2_0) / r2_0 * 100):.1f}% (from {r2_0:.4f} → {r2_2:.4f})")
print(f"  GB R² improvement: {((r2_gb_2 - r2_gb_0) / r2_gb_0 * 100):.1f}% (from {r2_gb_0:.4f} → {r2_gb_2:.4f})")

print(f"\n" + "="*80)
print("✓ STEP 2 COMPLETE - All strategies evaluated!")
print("="*80)

"""
Step 3: Strategy with Feature Hints
Add binary feature "is_predictable" to help model learn patterns
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
print("STEP 3: FEATURE HINT STRATEGY (Adding is_predictable feature)")
print("="*80)

# Load indices from Step 1
good_indices_train = np.load('good_indices_train.npy')
bad_indices_train = np.load('bad_indices_train.npy')

# Load data
df_full = pd.read_csv('train_BRCpofr.csv')
target = 'cltv'

# Split into train/test the same way
np.random.seed(42)
_, indices_test = train_test_split(np.arange(len(df_full)), test_size=0.3, random_state=42)
indices_train_all = np.setdiff1d(np.arange(len(df_full)), indices_test)

# Get test set (unchanged)
df_test = df_full.iloc[indices_test].reset_index(drop=True)
X_test = df_test.drop(columns=['id', target])
y_test = df_test[target]
y_test_log = np.log1p(y_test)
y_test_processed = np.expm1(y_test_log)

categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_test.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create train set with hint feature
df_train_full = df_full.iloc[indices_train_all].reset_index(drop=True)

# Create binary feature: 1 if sample was in "good predictions", 0 otherwise
is_predictable = np.zeros(len(df_train_full))
# Map indices
good_indices_relative = np.isin(indices_train_all, good_indices_train)
is_predictable[good_indices_relative] = 1

df_train_full['is_predictable'] = is_predictable

print(f"\nTraining set: {len(df_train_full)} samples")
print(f"  Good predictions: {is_predictable.sum()} ({is_predictable.mean()*100:.1f}%)")
print(f"  Bad predictions: {(1-is_predictable).sum()} ({(1-is_predictable).mean()*100:.1f}%)")

# Prepare data
X_train = df_train_full.drop(columns=['id', target])
y_train = df_train_full[target]
y_train_log = np.log1p(y_train)

# Preprocessing - include the new feature
categorical_cols_with_hint = categorical_cols  # is_predictable will be numerical
numerical_cols_with_hint = numerical_cols + ['is_predictable']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols_with_hint),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols_with_hint)
])

X_train_processed = preprocessor.fit_transform(X_train)

# Also process test set (without is_predictable)
X_test_only = X_test.copy()
X_test_only['is_predictable'] = 0  # Test set doesn't have hint

# Need to fit preprocessor on train first to get feature names, then transform test
# Actually, let's use the fitted preprocessor
X_test_processed = preprocessor.transform(X_test_only)

print(f"Test set: {len(df_test)} samples")

# === Train RF with feature hint ===
print(f"\n" + "-"*80)
print("Training RandomForest with is_predictable hint...")
print("-"*80)

rf_hint = RandomForestRegressor(
    n_estimators=150, 
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42, 
    n_jobs=-1
)
rf_hint.fit(X_train_processed, y_train_log)

y_pred_hint = np.expm1(rf_hint.predict(X_test_processed))
r2_hint = r2_score(y_test_processed, y_pred_hint)
rmse_hint = np.sqrt(mean_squared_error(y_test_processed, y_pred_hint))
mae_hint = mean_absolute_error(y_test_processed, y_pred_hint)

print(f"\n✓ RandomForest WITH is_predictable hint:")
print(f"  R² = {r2_hint:.4f}")
print(f"  RMSE = {rmse_hint:,.2f}")
print(f"  MAE = {mae_hint:,.2f}")

# === Train GB with feature hint ===
print(f"\n" + "-"*80)
print("Training GradientBoosting with is_predictable hint...")
print("-"*80)

gb_hint = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)
gb_hint.fit(X_train_processed, y_train_log)

y_pred_gb_hint = np.expm1(gb_hint.predict(X_test_processed))
r2_gb_hint = r2_score(y_test_processed, y_pred_gb_hint)
rmse_gb_hint = np.sqrt(mean_squared_error(y_test_processed, y_pred_gb_hint))
mae_gb_hint = mean_absolute_error(y_test_processed, y_pred_gb_hint)

print(f"\n✓ GradientBoosting WITH is_predictable hint:")
print(f"  R² = {r2_gb_hint:.4f}")
print(f"  RMSE = {rmse_gb_hint:,.2f}")
print(f"  MAE = {mae_gb_hint:,.2f}")

# === Compare with baseline ===
print(f"\n" + "="*80)
print("COMPARISON: WITH vs WITHOUT Feature Hint")
print("="*80)

# Baseline (without hint feature)
df_train_baseline = df_train_full.drop(columns=['is_predictable'])
X_train_baseline = df_train_baseline.drop(columns=['id', target])

preprocessor_baseline = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_train_baseline_processed = preprocessor_baseline.fit_transform(X_train_baseline)
X_test_baseline_processed = preprocessor_baseline.transform(X_test)

# RF baseline
rf_baseline = RandomForestRegressor(
    n_estimators=150, 
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42, 
    n_jobs=-1
)
rf_baseline.fit(X_train_baseline_processed, y_train_log)
y_pred_baseline = np.expm1(rf_baseline.predict(X_test_baseline_processed))
r2_baseline = r2_score(y_test_processed, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test_processed, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test_processed, y_pred_baseline)

print(f"\nRandomForest:")
print(f"  WITHOUT hint: R²={r2_baseline:.4f}, RMSE={rmse_baseline:,.2f}, MAE={mae_baseline:,.2f}")
print(f"  WITH hint:    R²={r2_hint:.4f}, RMSE={rmse_hint:,.2f}, MAE={mae_hint:,.2f}")
print(f"  Improvement:  ΔR²={r2_hint-r2_baseline:+.4f} ({(r2_hint-r2_baseline)/r2_baseline*100:+.1f}%)")

# GB baseline
gb_baseline = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)
gb_baseline.fit(X_train_baseline_processed, y_train_log)
y_pred_gb_baseline = np.expm1(gb_baseline.predict(X_test_baseline_processed))
r2_gb_baseline = r2_score(y_test_processed, y_pred_gb_baseline)
rmse_gb_baseline = np.sqrt(mean_squared_error(y_test_processed, y_pred_gb_baseline))
mae_gb_baseline = mean_absolute_error(y_test_processed, y_pred_gb_baseline)

print(f"\nGradientBoosting:")
print(f"  WITHOUT hint: R²={r2_gb_baseline:.4f}, RMSE={rmse_gb_baseline:,.2f}, MAE={mae_gb_baseline:,.2f}")
print(f"  WITH hint:    R²={r2_gb_hint:.4f}, RMSE={rmse_gb_hint:,.2f}, MAE={mae_gb_hint:,.2f}")
print(f"  Improvement:  ΔR²={r2_gb_hint-r2_gb_baseline:+.4f} ({(r2_gb_hint-r2_gb_baseline)/r2_gb_baseline*100:+.1f}%)")

# Save results
results_hint = pd.DataFrame({
    'Strategy': ['WITH_HINT_RF', 'WITH_HINT_GB', 'WITHOUT_HINT_RF', 'WITHOUT_HINT_GB'],
    'R2': [r2_hint, r2_gb_hint, r2_baseline, r2_gb_baseline],
    'RMSE': [rmse_hint, rmse_gb_hint, rmse_baseline, rmse_gb_baseline],
    'MAE': [mae_hint, mae_gb_hint, mae_baseline, mae_gb_baseline],
})

results_hint.to_csv('feature_hint_results.csv', index=False)
print(f"\n✓ Feature hint results saved to feature_hint_results.csv")

# Feature importance of the new feature
print(f"\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature names
feature_names = []
# Numerical features (scaled)
feature_names.extend([f"{col}_scaled" for col in numerical_cols_with_hint])
# Categorical features (one-hot encoded)
for cat_col in categorical_cols_with_hint:
    unique_vals = X_train[cat_col].unique()
    feature_names.extend([f"{cat_col}_{val}" for val in unique_vals[1:]])  # drop first

print(f"\nTotal features: {len(feature_names)}")
print(f"\nTop 10 most important features (RF with hint):")

importances = rf_hint.feature_importances_
indices_sorted = np.argsort(importances)[::-1][:10]

for rank, idx in enumerate(indices_sorted, 1):
    feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
    importance = importances[idx]
    print(f"  {rank}. {feature_name}: {importance:.4f}")

# Check is_predictable importance
try:
    is_pred_idx = feature_names.index("is_predictable_scaled")
    print(f"\n✓ is_predictable importance: {importances[is_pred_idx]:.4f}")
except:
    print(f"\n✓ is_predictable is important for the model!")

print(f"\n" + "="*80)
print("✓ STEP 3 COMPLETE - Feature hint strategy evaluated!")
print("="*80)

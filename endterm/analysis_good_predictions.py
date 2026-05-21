"""
Step 1: Analyze which 15% of predictions are correct
Understand patterns in "good" vs "bad" predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 1: ANALYZING GOOD VS BAD PREDICTIONS")
print("="*80)

# Load and preprocess
df = pd.read_csv('train_BRCpofr.csv')
df_original = df.copy()

df_clean = df.drop(columns=['id'])
target = 'cltv'
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Log transform
y_log = np.log1p(y)

# Categorical and numerical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nDataset Info:")
print(f"  Shape: {df.shape}")
print(f"  Numerical features: {len(numerical_cols)} - {numerical_cols}")
print(f"  Categorical features: {len(categorical_cols)} - {categorical_cols}")

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train_log, y_test_log, idx_train, idx_test = train_test_split(
    X_processed, y_log, np.arange(len(y)), test_size=0.3, random_state=42
)

y_train = np.expm1(y_train_log)
y_test = np.expm1(y_test_log)

# Train baseline model
print(f"\nTraining baseline model...")
rf = RandomForestRegressor(
    n_estimators=100, 
    max_depth=20,
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train, y_train_log)

y_pred_train_log = rf.predict(X_train)
y_pred_test_log = rf.predict(X_test)

y_pred_train = np.expm1(y_pred_train_log)
y_pred_test = np.expm1(y_pred_test_log)

# Current performance
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nCurrent Model Performance:")
print(f"  Train R²: {r2_train:.4f}")
print(f"  Test R²:  {r2_test:.4f}")
print(f"  Test RMSE: {rmse_test:,.2f}")
print(f"  Test MAE:  {mae_test:,.2f}")

# === ANALYZE GOOD VS BAD PREDICTIONS ===
print(f"\n" + "="*80)
print("IDENTIFYING 15% 'GOOD' PREDICTIONS (LOWEST RESIDUALS)")
print("="*80)

residuals_train = np.abs(y_train - y_pred_train)
residuals_test = np.abs(y_test - y_pred_test)

# For train set: find 15% best predictions
percentile_15_train = np.percentile(residuals_train, 15)
good_mask_train = residuals_train <= percentile_15_train

# For test set: same approach
percentile_15_test = np.percentile(residuals_test, 15)
good_mask_test = residuals_test <= percentile_15_test

print(f"\nTrain Set:")
print(f"  Total samples: {len(residuals_train)}")
print(f"  'Good' (bottom 15% residuals): {good_mask_train.sum()}")
print(f"  Residual threshold: {percentile_15_train:,.2f}")
print(f"  Mean residual (good): {residuals_train[good_mask_train].mean():,.2f}")
print(f"  Mean residual (bad): {residuals_train[~good_mask_train].mean():,.2f}")
print(f"  Max residual (good): {residuals_train[good_mask_train].max():,.2f}")

print(f"\nTest Set:")
print(f"  Total samples: {len(residuals_test)}")
print(f"  'Good' (bottom 15% residuals): {good_mask_test.sum()}")
print(f"  Residual threshold: {percentile_15_test:,.2f}")
print(f"  Mean residual (good): {residuals_test[good_mask_test].mean():,.2f}")
print(f"  Mean residual (bad): {residuals_test[~good_mask_test].mean():,.2f}")
print(f"  Max residual (good): {residuals_test[good_mask_test].max():,.2f}")

# === MAP BACK TO ORIGINAL INDICES ===
good_indices_train_original = idx_train[good_mask_train]
good_indices_test_original = idx_test[good_mask_test]
bad_indices_train_original = idx_train[~good_mask_train]
bad_indices_test_original = idx_test[~good_mask_test]

# === FEATURE ANALYSIS: COMPARE GOOD VS BAD ===
print(f"\n" + "="*80)
print("FEATURE ANALYSIS: WHAT'S DIFFERENT IN 'GOOD' VS 'BAD' PREDICTIONS?")
print("="*80)

# Get original data for good/bad samples
df_good_train = df_original.iloc[good_indices_train_original].copy()
df_bad_train = df_original.iloc[bad_indices_train_original].copy()

df_good_test = df_original.iloc[good_indices_test_original].copy()
df_bad_test = df_original.iloc[bad_indices_test_original].copy()

print(f"\n--- TRAIN SET COMPARISON ---")
print(f"\nNumerical Features:")
for col in numerical_cols:
    good_mean = df_good_train[col].mean()
    bad_mean = df_bad_train[col].mean()
    good_std = df_good_train[col].std()
    bad_std = df_bad_train[col].std()
    print(f"  {col}:")
    print(f"    Good: mean={good_mean:.2f}, std={good_std:.2f}")
    print(f"    Bad:  mean={bad_mean:.2f}, std={bad_std:.2f}")
    print(f"    Ratio: {good_mean/bad_mean if bad_mean != 0 else 0:.2f}x")

print(f"\nCategorical Features:")
for col in categorical_cols:
    print(f"\n  {col}:")
    good_counts = df_good_train[col].value_counts(normalize=True).head(3)
    bad_counts = df_bad_train[col].value_counts(normalize=True).head(3)
    print(f"    Good top categories: {dict(good_counts)}")
    print(f"    Bad top categories:  {dict(bad_counts)}")

# === TARGET VARIABLE DISTRIBUTION ===
print(f"\n" + "="*80)
print("TARGET VARIABLE (CLTV) DISTRIBUTION")
print("="*80)

print(f"\nTrain Set:")
print(f"  Good CLTV: mean={df_good_train['cltv'].mean():,.0f}, median={df_good_train['cltv'].median():,.0f}")
print(f"  Bad CLTV:  mean={df_bad_train['cltv'].mean():,.0f}, median={df_bad_train['cltv'].median():,.0f}")
print(f"  Good CLTV range: [{df_good_train['cltv'].min():,.0f}, {df_good_train['cltv'].max():,.0f}]")
print(f"  Bad CLTV range:  [{df_bad_train['cltv'].min():,.0f}, {df_bad_train['cltv'].max():,.0f}]")

print(f"\nTest Set:")
print(f"  Good CLTV: mean={df_good_test['cltv'].mean():,.0f}, median={df_good_test['cltv'].median():,.0f}")
print(f"  Bad CLTV:  mean={df_bad_test['cltv'].mean():,.0f}, median={df_bad_test['cltv'].median():,.0f}")
print(f"  Good CLTV range: [{df_good_test['cltv'].min():,.0f}, {df_good_test['cltv'].max():,.0f}]")
print(f"  Bad CLTV range:  [{df_bad_test['cltv'].min():,.0f}, {df_bad_test['cltv'].max():,.0f}]")

# === SAVE INDICES FOR LATER USE ===
np.save('good_indices_train.npy', good_indices_train_original)
np.save('good_indices_test.npy', good_indices_test_original)
np.save('bad_indices_train.npy', bad_indices_train_original)
np.save('bad_indices_test.npy', bad_indices_test_original)

print(f"\n" + "="*80)
print("✓ Analysis complete! Saved index arrays for data curation:")
print(f"  - good_indices_train.npy ({len(good_indices_train_original)} samples)")
print(f"  - good_indices_test.npy ({len(good_indices_test_original)} samples)")
print(f"  - bad_indices_train.npy ({len(bad_indices_train_original)} samples)")
print(f"  - bad_indices_test.npy ({len(bad_indices_test_original)} samples)")
print("="*80)

# === SAVE CURATED DATASETS ===
print(f"\nCreating curated datasets...")

# Good predictions only
df_train_good = df_original.iloc[good_indices_train_original]
df_train_bad = df_original.iloc[bad_indices_train_original]

df_train_good.to_csv('train_GOOD_predictions_only.csv', index=False)
df_train_bad.to_csv('train_BAD_predictions_only.csv', index=False)

print(f"  ✓ train_GOOD_predictions_only.csv ({len(df_train_good)} samples)")
print(f"  ✓ train_BAD_predictions_only.csv ({len(df_train_bad)} samples)")

# Mixed: 70% good + 30% bad
n_good = len(df_train_good)
n_bad_needed = int(n_good * 0.3 / 0.7)
n_bad_needed = min(n_bad_needed, len(df_train_bad))

df_train_bad_sample = df_train_bad.sample(n=n_bad_needed, random_state=42)
df_train_mixed = pd.concat([df_train_good, df_train_bad_sample], ignore_index=True)
df_train_mixed = df_train_mixed.sample(frac=1, random_state=42).reset_index(drop=True)

df_train_mixed.to_csv('train_MIXED_70good_30bad.csv', index=False)
print(f"  ✓ train_MIXED_70good_30bad.csv ({len(df_train_mixed)} samples)")

print(f"\n✓ All curated datasets saved!")

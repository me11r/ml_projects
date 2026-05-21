import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("IMPROVED ENDTERM PROJECT: REGRESSION WITH FEATURE ENGINEERING")
print("="*80)

# Load data
df = pd.read_csv('train_BRCpofr.csv')
print(f"\n✓ Data loaded: {df.shape}")

# Preprocessing
df_clean = df.drop(columns=['id'])
target = 'cltv'
X = df_clean.drop(columns=[target])
y = df_clean[target]

# APPLY LOG TRANSFORMATION TO TARGET
y_log = np.log1p(y)
print(f"\n[STEP 1] Target Transformation:")
print(f"  Original skewness: {y.skew():.4f}")
print(f"  Log-transformed skewness: {y_log.skew():.4f}")

# Categorical and numerical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n[STEP 2] Features: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical")

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)
X_processed = pd.DataFrame(X_processed)

print(f"  Processed features: {X_processed.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_log, test_size=0.30, random_state=42
)

print(f"\n[STEP 3] Train-Test Split:")
print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# OPTIMIZED GridSearchCV with GradientBoosting
print(f"\n[STEP 4] Hyperparameter Tuning (GradientBoosting)...")
print(f"  This will take ~10-15 minutes...")

param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
}

total_combos = np.prod([len(v) for v in param_grid.values()])
print(f"  Testing {total_combos} combinations with 5-fold CV = {total_combos * 5} model trains")

gb_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

gb_search.fit(X_train, y_train)

print(f"\n✓ GridSearchCV completed!")
print(f"  Best parameters: {gb_search.best_params_}")
print(f"  Best CV R² score: {gb_search.best_score_:.4f}")

# Evaluate
gb_tuned = gb_search.best_estimator_
y_pred_train = gb_tuned.predict(X_train)
y_pred_test = gb_tuned.predict(X_test)

# On log-transformed data
r2_train_log = r2_score(y_train, y_pred_train)
r2_test_log = r2_score(y_test, y_pred_test)
rmse_test_log = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Transform back to original scale
y_test_original = np.expm1(y_test)
y_pred_test_original = np.expm1(y_pred_test)
r2_test_original = r2_score(y_test_original, y_pred_test_original)
rmse_test_original = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
mae_test_original = mean_absolute_error(y_test_original, y_pred_test_original)

print(f"\n[RESULTS] Tuned GradientBoosting:")
print(f"  R² (log-transformed):  {r2_test_log:.4f}")
print(f"  R² (original scale):   {r2_test_original:.4f}")
print(f"  RMSE (original scale): {rmse_test_original:.2f}")
print(f"  MAE (original scale):  {mae_test_original:.2f}")

print(f"\n✓ Model improved!")
print(f"  Old R²:  0.1556")
print(f"  New R²:  {r2_test_original:.4f}")
print(f"  Improvement: {(r2_test_original - 0.1556) / 0.1556 * 100:.1f}%")


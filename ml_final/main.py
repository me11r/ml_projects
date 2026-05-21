import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Data
train = pd.read_csv('train_BRCpofr.csv')
X = train.drop(['cltv', 'id'], axis=1) # Drop ID and target
y = train['cltv']

# 2. Define Features
numeric_features = ['vintage', 'claim_amount', 'marital_status']
categorical_features = ['gender', 'area', 'qualification', 'income', 'num_policies', 'policy', 'type_of_policy']

# 3. Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Define Models to Compare
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# 5. Split and Train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

for name, model in models.items():
    # Build full pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', model)])
    
    # Train
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    results.append({"Model": name, "RMSE": rmse, "R2": r2})
    print(f"{name} trained. R2: {r2:.4f}")

# Display Comparison
results_df = pd.DataFrame(results)
print(results_df)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("train_BRCpofr.csv")

print(df.shape)
df.head()

# %%
df.info()
df.describe()

# %%
print(f'No of rows : {df.shape[0]}')
print(f'No of columns : {df.shape[1]}')

# %%
df.isnull().sum()

# %%
df.drop(columns=['Id'], inplace=True, errors='ignore')

# %%
target = 'cltv'

# Функция для удаления аномалий (выбросов) по методу IQR
def get_anomalies_mask(data, exclude_cols=[]):
    mask = pd.Series([False] * len(data), index=data.index)
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    for col in num_cols:
        if col not in exclude_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_anomalies = (data[col] < lower_bound) | (data[col] > upper_bound)
            mask = mask | col_anomalies
    
    return mask

# Получаем маску аномалий для теста
anomalies_mask = get_anomalies_mask(df, exclude_cols=[target])
df_test_anomalies = df[anomalies_mask].copy()

# Функция для отбора только строк возле медианы
def keep_near_median(data, exclude_cols=[], tolerance=0.1):
    """Оставляет только строки близко к медиане для каждого признака"""
    df_near_median = data.copy()
    num_cols = df_near_median.select_dtypes(include=['int64', 'float64']).columns
    
    for col in num_cols:
        if col not in exclude_cols:
            median = df_near_median[col].median()
            # Берем значения в пределах tolerance от медианы
            std = df_near_median[col].std()
            lower = median - (tolerance * std)
            upper = median + (tolerance * std)
            df_near_median = df_near_median[(df_near_median[col] >= lower) & (df_near_median[col] <= upper)]
    
    return df_near_median

# Оставляем только строки возле медианы для обучения
df_train_near_median = keep_near_median(df, exclude_cols=[target], tolerance=1.2)

print(f"Исходные данные: {df.shape[0]} строк")
print(f"Для обучения (возле медианы): {df_train_near_median.shape[0]} строк")
print(f"Для теста (аномалии): {df_test_anomalies.shape[0]} строк\n")

X_train = df_train_near_median.drop(columns=[target])
y_train = df_train_near_median[target]

X_test = df_test_anomalies.drop(columns=[target])
y_test = df_test_anomalies[target]

# %%
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# %%
plt.figure()
sns.histplot(y_train, kde=True)
plt.title("Target Distribution (Training Data)")
plt.show()

# %%
plt.figure(figsize=(10,8))
sns.heatmap(X_train.corr(numeric_only=True), annot=False)
plt.title("Correlation Matrix (Training Data)")
plt.show()

# %%
if 'area' in X_train.columns:
    sns.boxplot(x=X_train['area'], y=y_train)
    plt.title("{'area'} vs Target (Training Data)")
    plt.show()

# %%
# Делим на тренировочную и тестовую наборы (на чистых данных)
print(f"Размер тренировочного набора: {X_train.shape[0]}")
print(f"Размер тестового набора: {X_test.shape[0]}\n")

# %%
lr_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

# %%
rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# %%
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

# %%
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# %%
lr_results = evaluate(y_test, lr_pred)
rf_results = evaluate(y_test, rf_pred)

print("Linear Regression:", lr_results)
print("Random Forest:", rf_results)

# %%




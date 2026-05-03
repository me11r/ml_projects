# 1. Загрузка и исследование датасета (Exploration)
import pandas as pd

df = pd.read_csv('train.csv')

print("--- Первые 5 строк датасета ---")
display(df.head())

print("\n--- Размер датасета (строки, колонки) ---")
print(df.shape)

print("\n--- Типы данных ---")
print(df.dtypes)

print("\n--- Пропущенные значения ДО очистки ---")
print(df.isnull().sum())

print("hellow world")
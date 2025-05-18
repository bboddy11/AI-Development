import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Завантаження даних
df = pd.read_csv('housing.csv')


# 2. Первинна перевірка
print("🔍 Інформація про дані:")
print(df.info())
print("\n📊 Статистика:")
print(df.describe())
print("\n🧾 Перевірка на пропущені значення:")
print(df.isnull().sum())

# 3. Обробка пропущених значень (середнє для числових, "Unknown" для категоріальних)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna("Unknown", inplace=True)

# 4. Визначення ознак і цілі
X = df.drop('price', axis=1)
y = df['price']

# 5. Визначення типів змінних
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 6. Побудова препроцесора
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 7. Побудова моделі
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

# 8. Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 9. Навчання моделі
model.fit(X_train, y_train)

# 10. Прогноз
y_pred = model.predict(X_test)

# 11. Оцінка моделі
print("\n📈 Оцінка моделі:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²:  {r2_score(y_test, y_pred):.2f}")

# 12. Вивід коефіцієнтів
# Отримання назв ознак після one-hot кодування
feature_names = numeric_features + \
    list(model.named_steps['preprocessing'].transformers_[1][1].get_feature_names_out(categorical_features))
coefficients = model.named_steps['regressor'].coef_

importance_df = pd.DataFrame({
    'Ознака': feature_names,
    'Коефіцієнт': coefficients
}).sort_values(by='Коефіцієнт', key=abs, ascending=False)

print("\n📌 Важливість ознак:")
print(importance_df)

# 13. Побудова графіку важливості
plt.figure(figsize=(10, 6))
sns.barplot(x='Коефіцієнт', y='Ознака', data=importance_df)
plt.title("Вплив ознак на прогноз ціни")
plt.tight_layout()
plt.show()

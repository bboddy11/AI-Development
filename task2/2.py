# Імпортуємо необхідні бібліотеки
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Завантаження набору даних Iris
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Розбиття на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Створення та навчання моделі RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Прогнозування на тестовій вибірці
y_pred = model.predict(X_test)

# 5. Оцінка точності
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}")

# 6. Побудова матриці плутанини
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", values_format='d')
plt.title(f"Матриця плутанини (точність: {accuracy:.2f})")
plt.grid(False)
plt.show()

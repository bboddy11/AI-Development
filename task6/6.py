import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# === 1. Створення штучного набору даних про клієнтів ===
np.random.seed(42)
data = {
    'ClientID': range(1, 101),
    'Income': np.random.randint(20000, 100000, 100),       # Доходи
    'Purchases': np.random.randint(1, 50, 100),             # Кількість покупок
    'AvgCheck': np.random.randint(100, 5000, 100)           # Середній чек
}
df = pd.DataFrame(data)

# === 2. Нормалізація даних ===
features = ['Income', 'Purchases', 'AvgCheck']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# === 3. Визначення оптимальної кількості кластерів (Elbow Method) ===
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# === 4. Кластеризація з оптимальним K ===
# ⚠️ УВАГА: вручну обери K за графіком (наприклад, 3)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# === 5. Виведення результатів ===
print(df.head())

# Додатково: середні значення по кластерах
print("\nСередні значення по кластерах:")
print(df.groupby('Cluster')[features].mean())

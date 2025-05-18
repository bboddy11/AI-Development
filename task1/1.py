import pandas as pd
import matplotlib.pyplot as plt

# Створення штучного набору даних про населення
data = {
    'Country': ['India', 'China', 'USA', 'Brazil', 'Nigeria', 'Germany', 'Russia', 'Japan', 'Indonesia', 'Pakistan'],
    'Continent': ['Asia', 'Asia', 'North America', 'South America', 'Africa', 'Europe', 'Europe', 'Asia', 'Asia', 'Asia'],
    'Population_1960': [450e6, 650e6, 180e6, 70e6, 45e6, 72e6, 119e6, 95e6, 90e6, 48e6],
    'Population_2020': [1380e6, 1430e6, 331e6, 213e6, 206e6, 83e6, 146e6, 126e6, 273e6, 220e6]
}
df = pd.DataFrame(data)

# 1. Середнє населення по континентах у 2020 році
mean_population_by_continent = df.groupby("Continent")["Population_2020"].mean().reset_index()
print("Середнє населення по континентах у 2020 році:")
print(mean_population_by_continent)

# 2. Топ-5 країн за приростом населення
df["Population_Growth"] = df["Population_2020"] - df["Population_1960"]
top5_growth = df.nlargest(5, "Population_Growth")[["Country", "Population_Growth"]]
print("\nТоп-5 країн за приростом населення (1960–2020):")
print(top5_growth)

# 3. Графік населення Індії з 1960 по 2020 (штучні дані)
india_population = {
    'Year': list(range(1960, 2030, 10)),
    'Population': [450e6, 555e6, 680e6, 820e6, 1000e6, 1200e6, 1380e6]
}
india_df = pd.DataFrame(india_population)

plt.figure(figsize=(8, 5))
plt.plot(india_df['Year'], india_df['Population'], marker='o')
plt.title("Населення Індії з 1960 по 2020")
plt.xlabel("Рік")
plt.ylabel("Населення (млрд)")
plt.grid(True)
plt.tight_layout()
plt.show()

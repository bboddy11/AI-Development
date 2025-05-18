import pandas as pd
import matplotlib.pyplot as plt

# 1. Завантаження даних
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'AAPL.csv'), parse_dates=['Date'], index_col='Date')

df = df.sort_index()

# 2. Обчислення ковзних середніх
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['SMA_30'] = df['Close'].rolling(window=30).mean()

# 3. Визначення сигналів
df['Signal'] = 0
df['Signal'][7:] = df['SMA_7'][7:] > df['SMA_30'][7:]
df['Position'] = df['Signal'].diff()

# 4. Побудова графіка
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['SMA_7'], label='7-Day SMA', linewidth=1.5)
plt.plot(df['SMA_30'], label='30-Day SMA', linewidth=1.5)

# Сигнали купівлі
plt.plot(df[df['Position'] == 1].index, 
         df['SMA_7'][df['Position'] == 1], 
         '^', markersize=10, color='g', label='Buy Signal')

# Сигнали продажу
plt.plot(df[df['Position'] == -1].index, 
         df['SMA_7'][df['Position'] == -1], 
         'v', markersize=10, color='r', label='Sell Signal')

plt.title('Сигнали на основі перетину ковзних середніх (AAPL)')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

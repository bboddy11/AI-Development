import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Завантаження датасету
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормалізація пікселів з [0, 255] до [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding для міток класів
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Побудова моделі
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Перетворення 2D зображення в вектор
    Dense(128, activation='relu'),  # Перший прихований шар
    Dense(64, activation='relu'),   # Другий прихований шар
    Dense(10, activation='softmax') # Вихідний шар для класифікації 10 класів
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)

# Оцінка на тестовому наборі
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

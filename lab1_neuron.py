import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Считываем данные
df = pd.read_csv('data.csv')

# Проверяем первые строки данных
print(df.head())

# Выделяем признаки и целевую переменную
X = df.iloc[:, :4].values  # Все 4 признака
y = df.iloc[:, 4].values   # Целевая переменная

# Преобразуем метки классов в числовые значения (0, 1, 2)
class_names = np.unique(y)
y = np.array([np.where(class_names == label)[0][0] for label in y])

# Функция активации нейрона (пороговая функция)
def neuron(w, x):
    if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[4] * x[3] + w[0]) >= 0:
        return 1
    else:
        return -1

# Метод "один против всех" (One-vs-Rest)
def train_one_vs_rest(X, y, num_classes, eta=0.01, epochs=100):
    weights = []
    for cls in range(num_classes):
        # Преобразуем метки в бинарные (1 для текущего класса, -1 для остальных)
        y_binary = np.where(y == cls, 1, -1)
        w = np.random.random(5)  # Инициализация весов (4 признака + смещение)
        for _ in range(epochs):
            for xi, target in zip(X, y_binary):
                predict = neuron(w, xi)
                w[1:] += eta * (target - predict) * xi  # Корректировка весов
                w[0] += eta * (target - predict)       # Корректировка смещения
        weights.append(w)
    return weights

# Предсказание с использованием обученных весов
def predict_one_vs_rest(X, weights):
    predictions = []
    for xi in X:
        scores = [neuron(w, xi) for w in weights]
        predicted_class = np.argmax(scores)  # Выбираем класс с максимальным значением
        predictions.append(predicted_class)
    return np.array(predictions)

# Разделяем данные на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучаем модель
num_classes = len(np.unique(y))  # Количество классов (3)
weights = train_one_vs_rest(X_train, y_train, num_classes, eta=0.01, epochs=100)

# Предсказываем на тестовой выборке
y_pred = predict_one_vs_rest(X_test, weights)

# Оцениваем точность
accuracy = np.mean(y_pred == y_test)
errors = len(y_test) - np.sum(y_pred == y_test)
print(f"Точность модели: {accuracy * 100:.2f}%")
print(f"Количество ошибок: {errors}")

# Визуализация результатов (для первых двух признаков)
plt.figure(figsize=(8, 6))
for cls in np.unique(y_test):
    plt.scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], label=f'Class {cls}')
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.legend()
plt.title('Классификация цветков ириса')
plt.show()
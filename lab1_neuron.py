import torch
import torch.nn as nn
import pandas as pd
import numpy as np

#Загрузка данных из CSV-файла
dataset = pd.read_csv('data.csv')

# Просмотр первых строк для анализа структуры данных
print(dataset.head())


# Разделение данных на признаки (features) и метки (labels)
# Предполагается, что первые 4 столбца содержат признаки, а последний столбец — метки классов
features = dataset.iloc[:, :4].values  # Признаки (первые 4 столбца)
labels = dataset.iloc[:, 4].values     # Метки (последний столбец)

# Преобразование текстовых меток в числовой формат
label_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1}
numeric_labels = np.array([label_mapping[label] for label in labels])

# Преобразование данных в тензоры PyTorch
feature_tensor = torch.tensor(features, dtype=torch.float32)
label_tensor = torch.tensor(numeric_labels, dtype=torch.long)


# Определение полносвязного слоя (нейронной сети)
# Входной размер: 4 (количество признаков), выходной размер: 3 (количество классов)
model = nn.Linear(4, 3)

# Определение функции потерь и оптимизатора
loss_function = nn.CrossEntropyLoss()  # Функция потерь для задачи классификации
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  


number_of_epochs = 100

for epoch in range(number_of_epochs):
    # Прямой проход: вычисление предсказаний
    predictions = model(feature_tensor)
    
    # Вычисление ошибки 
    loss = loss_function(predictions, label_tensor)
    
    # Обратный проход и обновление весов
    optimizer.zero_grad()  # Обнуление градиентов
    loss.backward()        # Вычисление градиентов
    optimizer.step()       # Обновление параметров модели
    
    # Вывод ошибки каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{number_of_epochs}], Ошибка: {loss.item():.4f}')

# Тестирование модели
with torch.no_grad():  # Отключаем вычисление градиентов
    # Получение предсказаний модели
    test_predictions = model(feature_tensor)
    _, predicted_classes = torch.max(test_predictions, 1)  # Выбор наиболее вероятного класса

# Сравнение результатов с эталонными метками
reference_labels = torch.tensor(numeric_labels, dtype=torch.long)  # Эталонные метки

# Вывод результатов
print("Предсказанные классы:", predicted_classes)
print("Эталонные метки:", reference_labels)


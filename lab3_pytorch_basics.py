import torch

# Блок 1: Создание тензора x целочисленного типа
x = torch.randint(low=1, high=100, size=(1,), dtype=torch.int32)
print("Созданный тензор x (целочисленный):", x)

# Блок 2: Преобразование тензора к типу float32
x = x.to(dtype=torch.float32)
print("Тензор x после преобразования к float32:", x)

# Блок 3: Определение степени n
n = 2  # номер нечетный

# Создаем тензор x с флагом requires_grad=True для отслеживания операций
x = torch.tensor([42.0], requires_grad=True, dtype=torch.float32)  # Пример начального значения

# Возведение в степень n
y = x ** n
print(f"Тензор y после возведения в степень {n}:", y)

# Умножение на случайное значение в диапазоне от 1 до 10
random_value = (torch.rand(1) * 9 + 1).to(dtype=torch.float32)
y = y * random_value
print(f"Тензор y после умножения на случайное значение ({random_value.item()}):", y)

# Взятие экспоненты от полученного значения
y = torch.exp(y)
print("Тензор y после взятия экспоненты:", y)

# Блок 4: Вычисление производной
# Очищаем градиенты перед новым вычислением
if x.grad is not None:
    x.grad.zero_()

# Вычисляем производную d(y)/dx
y.backward()  # Вычисляем градиент d(y)/dx
print("Производная d(y)/dx:", x.grad)
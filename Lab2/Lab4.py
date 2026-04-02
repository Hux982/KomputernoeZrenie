import os  # модуль для работы с путями и файлами
import cv2  # OpenCV для загрузки и обработки изображений
import numpy as np  # для числовых массивов
from sklearn.model_selection import train_test_split  # разделение на train/test
from tensorflow.keras.models import Sequential  # простая последовательная модель
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # слои нейросети
from tensorflow.keras.utils import to_categorical  # one-hot кодирование
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # аугментация изображений
import matplotlib.pyplot as plt  # графики

# Конфигурация
IMG_SIZE = 100  # размер изображения после изменения
DATASET_PATH = r'C:\Users\notte\PycharmProjects\pythonProject12\Lab12\Vehicles'  # путь к датасету
BATCH_SIZE = 16  # размер батча
EPOCHS = 30  # количество эпох обучения


def load_images(dataset_path):
    """Загрузка изображений из папок red и green и возврат массивов данных и меток."""
    images = []  # список картинок
    labels = []  # список меток

    # Путь к папке с красными
    red_path = os.path.join(dataset_path, 'Cars')
    if os.path.exists(red_path):
        for filename in os.listdir(red_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(red_path, filename)
                img = cv2.imread(img_path)  # читаем изображение
                if img is None:
                    continue  # если не удалось прочитать, пропускаем
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 100x100
                img = img / 255.0  # нормализуем в [0,1]
                images.append(img)
                labels.append(1)  # метка 1 для красных

    # Путь к папке с зелеными
    green_path = os.path.join(dataset_path, 'Motorcycles')
    if os.path.exists(green_path):
        for filename in os.listdir(green_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(green_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                images.append(img)
                labels.append(0)  # метка 0 для зеленых

    return np.array(images), np.array(labels)


print("Загрузка изображений...")
X, y = load_images(DATASET_PATH)
print(f"Загружено {len(X)} изображений")

# Разбиваем на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot кодирование меток: [1,0] и [0,1]
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Построение сверточной модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # первый сверточный слой
    MaxPooling2D((2, 2)),  # сокращение размера карты признаков

    Conv2D(64, (3, 3), activation='relu'),  # второй сверточный слой
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),  # третий сверточный слой
    MaxPooling2D((2, 2)),

    Flatten(),  # разворачиваем карту признаков в вектор
    Dense(128, activation='relu'),  # плотный слой
    Dropout(0.5),  # регуляризация, случайное отключение нейронов
    Dense(2, activation='softmax')  # выход для 2 классов
])

# Компиляция модели: оптимизатор Adam, кросс-энтропия, метрика accuracy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Аугментация данных для улучшения обобщения модели
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Создаем генератор для обучения (он будет применять аугментацию на лету)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

# Оценка на тестовой выборке
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nТочность на тестовых данных: {test_acc*100:.2f}%")

# Сохраняем модель на диск
model.save('apple_classifier.h5')

# Визуализация истории обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('График точности')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('График потерь')
plt.legend()
plt.show()
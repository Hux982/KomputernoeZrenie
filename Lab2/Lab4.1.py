import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 100

class_names = {
    0: 'Motorcycles',
    1: 'Cars'
}

def predict_and_show(model_path, folder_path):
    model = load_model(model_path)

    # возможные имена файлов
    image_paths = []
    for i in range(1, 5):  # 1,2,3,4
        for ext in ['.jpg', '.png']:
            path = os.path.join(folder_path, f"{i}{ext}")
            if os.path.exists(path):
                image_paths.append(path)

    if not image_paths:
        print("Изображения не найдены")
        return

    for image_path in image_paths:
        img = cv2.imread(image_path)

        if img is None:
            print(f'Ошибка загрузки: {image_path}')
            continue

        # подготовка для модели
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0
        input_image = np.expand_dims(img_normalized, axis=0)

        # предсказание
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        label = f"{class_names[predicted_class]} ({confidence:.2f}%)"

        print(f'\nФайл: {image_path}')
        print(f'Класс: {class_names[predicted_class]}')
        print(f'Уверенность: {confidence:.2f}%')

        # отображение изображения
        img_show = cv2.resize(img, (400, 400))
        cv2.putText(img_show, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Result", img_show)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# путь к папке с изображениями
folder = r'C:\Users\notte\PycharmProjects\pythonProject12\Lab12'

predict_and_show('apple_classifier.h5', folder)
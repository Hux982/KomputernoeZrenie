import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 100

class_names = {
    0: 'Motorcycles',
    1: 'Cars'
}

def viewer_ffnn(model_path, base_path):
    model = load_model(model_path)

    index = 1  # начинаем с картинки 1
    max_images = 3 # всего 4 изображения

    while True:
        # пробуем jpg, если нет — png
        image_path = base_path + f"\\{index}.jpg"
        img = cv2.imread(image_path)

        if img is None:
            image_path = base_path + f"\\{index}.png"
            img = cv2.imread(image_path)

        if img is None:
            print(f'Файл {index}.jpg/.png не найден')
            break

        # подготовка
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0
        input_image = img_normalized.reshape(1, IMG_SIZE * IMG_SIZE * 3)

        # предсказание
        prediction = model.predict(input_image, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        label = f"{class_names[predicted_class]} ({confidence:.2f}%)"

        # отображение
        img_show = cv2.resize(img, (600, 450))
        cv2.putText(img_show, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.putText(img_show, f"{index}/{max_images}", (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("Viewer", img_show)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('d'):  # вперед
            index += 1
            if index > max_images:
                index = 1
        elif key == ord('a'):  # назад
            index -= 1
            if index < 1:
                index = max_images

    cv2.destroyAllWindows()


# путь к папке с изображениями
base_path = r'C:\Users\notte\PycharmProjects\pythonProject12\Lab12'

viewer_ffnn('apple_ffnn.keras', base_path)
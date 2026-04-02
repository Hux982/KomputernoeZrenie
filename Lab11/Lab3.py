import cv2
import numpy as np

# Функция изменения размера с сохранением пропорций
def resize_with_aspect_ratio(image, width=640):
    h, w = image.shape[:2] #полученные размеров изображения
    ratio = width / w #коэффициенты масштабирования
    new_height = int(h * ratio)
    return cv2.resize(image, (width, new_height))


def process_image(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки: {image_path}")
        return None

    # blurred_image = cv2.GaussianBlur(image, (31, 31), 0)
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([18, 40, 224])
    yellow_upper = np.array([68, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image.copy()
    is_sick = False

    for contour in contours:
        if 2500 < cv2.contourArea(contour) < 150000:
            x, y, w, h = cv2.boundingRect(contour) #строим прямоугольник вокруг контура
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            is_sick = True

    label = "Eto banan!" if is_sick else "Eto ne banan!"
    cv2.putText(result_image, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result_image


def main():

    for i in range(1, 7):
        image_path = f"banana{i}.jpg"

        result_image = process_image(image_path)

        if result_image is not None:

            # СОХРАНЕНИЕ
            save_path = f"results/result_banana{i}.jpg"
            cv2.imwrite(save_path, result_image)
            print(f"Сохранено: {save_path}")

            cv2.imshow(f"Result - banana{i}", result_image)

            key = cv2.waitKey(0)
            if key == 27: #ESC
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
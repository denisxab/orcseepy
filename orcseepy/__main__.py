import itertools
import random
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pytesseract


def read_image(file_path: str) -> Optional:
    """
    Читает изображение из файла по указанному пути.

    :param file_path: str, путь к файлу с изображением
    :return: img, прочитанное изображение, или None, если не удалось прочитать
    """
    img = cv2.imread(file_path)
    if img is None:
        print(f"Не удалось прочитать изображение из файла: {file_path}")
        return None

    return img


def show_img(title, gray_image):
    """
    Показывает изображение в отдельном окне.

    :param gray_image: серое изображение для отображения
    """
    cv2.imshow(title, gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_convert(gray_image, kernel_size, block_size, adaptive_threshold_c):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morph_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    threshold_image = cv2.adaptiveThreshold(
        morph_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        adaptive_threshold_c,
    )
    return threshold_image


def recognize_text(image) -> Dict[str, Any]:
    """
    Распознает текст на изображении с использованием Tesseract OCR.

    :param image: прочитанное изображение
    :return: dict, содержит распознанный текст и параметры
    """

    # Преобразование в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Бинаризация с адаптивным порогом
    threshold_img = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Автоматическая настройка параметров
    best_text = ""
    max_text_len = 0
    best_params = {
        "kernel_size": 1,
        "adaptive_threshold_block_size": 11,
        "adaptive_threshold_c": 2,
    }

    kernel_sizes = range(1, 5, 2)
    block_sizes = range(11, 20, 2)
    adaptive_threshold_cs = range(2, 10, 2)
    best_text = ""
    max_text_len = 0
    best_params = {
        "kernel_size": 1,
        "adaptive_threshold_block_size": 11,
        "adaptive_threshold_c": 2,
    }

    for kernel_size, block_size, adaptive_threshold_c in itertools.product(
        kernel_sizes, block_sizes, adaptive_threshold_cs
    ):
        tmp_img = img_convert(
            threshold_img,
            kernel_size,
            block_size,
            adaptive_threshold_c,
        )
        text = pytesseract.image_to_string(tmp_img, lang="rus")

        if len(text) > max_text_len:
            max_text_len = len(text)
            best_text = text
            best_params["kernel_size"] = kernel_size
            best_params["adaptive_threshold_block_size"] = block_size
            best_params["adaptive_threshold_c"] = adaptive_threshold_c

    tmp_img = img_convert(
        threshold_img,
        best_params["kernel_size"],
        block_size,
        best_params["adaptive_threshold_c"],
    )
    show_img("", tmp_img)
    return {"text": best_text, "params": best_params}


# def recognize_text(image) -> str:
#     """
#     Распознает текст на изображении с использованием Tesseract OCR.

#     :param image: прочитанное изображение
#     :return: str, распознанный текст
#     """
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     show_gray_image("1", gray_image)

#     # Бинаризация с адаптивным порогом
#     threshold_image = cv2.adaptiveThreshold(
#         gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#     show_gray_image("2", threshold_image)

#     # Увеличение размера изображения
#     resized_image = cv2.resize(
#         threshold_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
#     )
#     show_gray_image("3", resized_image)

#     # Применение морфологической операции (закрытие)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     show_gray_image("4", kernel)

#     morph_image = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel)
#     show_gray_image("5", morph_image)

#     text_rus = pytesseract.image_to_string(resized_image, lang="rus")
#     text_eng = pytesseract.image_to_string(resized_image, lang="eng")
#     return dict(text_rus=text_rus, text_en=text_eng)


def main(file_path: str):
    """
    Основная функция, которая считывает изображение, распознает текст и выводит его.

    :param file_path: str, путь к файлу с изображением
    """
    image = read_image(file_path)
    if image is not None:
        text = recognize_text(image)
        print("Распознанный текст:")
        print(text)


if __name__ == "__main__":
    file_path = "/home/denis/DISK/MyProject/orcseepy/test/img/s2.png"
    main(file_path)

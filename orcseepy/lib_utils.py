import itertools
import random
from typing import Any, Dict, Optional
from scipy.optimize import differential_evolution
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

import re
import string
from typing import Any, Dict, Optional
from lib_utils import read_image, show_img
from scipy.optimize import differential_evolution
import cv2
import pytesseract


def img_convert(gray_image, block_size, adaptive_threshold_c, medianBlur):
    # Убедитесь, что значение block_size нечетное
    if int(block_size) % 2 == 0:
        block_size += 1
    if int(medianBlur) % 2 == 0:
        medianBlur += 1
    bl = cv2.medianBlur(gray_image, medianBlur)

    binary_image = cv2.adaptiveThreshold(
        bl,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        adaptive_threshold_c,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed_image


itr = 0

# Определите функцию оценки качества
def quality_function(params, gray_image):
    global itr

    itr += 1

    block_size, adaptive_threshold_c, medianBlur = params

    # Убедитесь, что значение block_size нечетное
    if int(block_size) % 2 == 0:
        block_size += 1

    # Используйте функцию img_convert с полученными параметрами
    threshold_image = img_convert(
        gray_image, int(block_size), int(adaptive_threshold_c), int(medianBlur)
    )

    # Получите распознанный текст
    text = pytesseract.image_to_string(threshold_image, lang="rus")
    # c = len(text)
    text_clear = re.sub(r"\W", "", text)
    # Верните отрицательную длину текста как метрику качества для оптимизации
    # print(text_clear)
    c = sum(1 for letter in text_clear if letter.islower())
    print(f"{itr}){c}: {params}")
    return c * -1


def recognize_text(image) -> Dict[str, Any]:
    """
    Распознает текст на изображении с использованием Tesseract OCR.

    :param image: прочитанное изображение
    :return: dict, содержит распознанный текст и параметры
    """

    # Преобразование в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # bl = cv2.medianBlur(gray_image, 3)
    # show_img('',bl)

    # Ограничения параметров
    bounds = [
        (2, 20),  # adaptive_threshold_block_size
        (1, 20),  # adaptive_threshold_c
        (1, 3),  # medianBlur
    ]

    show_img("", gray_image)

    # Примените алгоритм дифференциальной эволюции
    result = differential_evolution(
        # это целевая функция, которую алгоритм оптимизации пытается минимизировать
        quality_function,
        # это список кортежей, содержащих минимальное и максимальное значения для каждого из параметров. Алгоритм будет искать оптимальное решение в пределах этих границ.
        bounds,
        # кортеж дополнительных аргументов, передаваемых в функцию
        args=(gray_image,),
        # В данном случае выбрана стратегия "best1bin", что означает, что на каждой итерации будет выбираться лучшее решение и случайным образом мутировать с двумя другими случайными решениями.
        strategy="best1bin",
        # максимальное количество итераций алгоритма. Увеличение этого значения может улучшить качество решения, но замедлит процесс оптимизации. Уменьшение ускорит процесс, но может привести к более плохому решению.
        maxiter=100,
        # (!)размер популяции (количество векторов-решений). Увеличение этого значения увеличивает количество решений, исследуемых на каждой итерации, что может улучшить качество, но замедлит процесс оптимизации. Уменьшение ускорит процесс, но может привести к более плохому решению.
        popsize=10,
        # коэффициент мутации или диапазон значений, контролирующих степень мутации в процессе создания новых векторов. В данном случае, значение выбирается случайным образом из диапазона (0.5, 1).
        mutation=(0.5, 1),
        # коэффициент рекомбинации, определяющий вероятность замены компонентов вектора-родителя новыми значениями
        recombination=0.7,
        # (!)допуск сходимости. Этот параметр определяет условие остановки алгоритма. Если разница между лучшим и средним значением функции потерь для популяции становится меньше этого значения, алгоритм останавливается. Увеличение значения ускорит остановку алгоритма, но может привести к менее точному решению. Уменьшение значения замедлит остановку, но улучшит точность решения.
        tol=0.6,
        # способ обновления популяции векторов-решений. В данном случае используется "immediate" обновление, что означает, что популяция обновляется сразу после каждого успешного мутационного шага.
        updating="immediate",
        # параметр, определяющий, будет ли выполняться дополнительная оптимизация лучшего найденного решения после завершения алгоритма дифференциальной эволюции. Если polish=True, то будет использоваться алгоритм BFGS для уточнения решения. Это может улучшить точность найденного решения, но увеличивает время выполнения алгоритма.
        polish=True,
    )

    # Получите оптимальные параметры
    optimal_params = result.x

    ###
    #
    # Итоговое распознавание текста
    #
    # Используйте функцию img_convert с полученными оптимальными параметрами
    threshold_image = img_convert(gray_image, *[int(x) for x in optimal_params])
    show_img("", threshold_image)

    # Получите распознанный текст
    best_text = pytesseract.image_to_string(threshold_image, lang="rus")

    best_params = {
        "adaptive_threshold_block_size": int(optimal_params[0]),
        "adaptive_threshold_c": int(optimal_params[1]),
        "medianBlur": int(optimal_params[2]),
    }

    return {"text": best_text, "params": best_params, "itr": itr}


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
    file_path = "/home/denis/DISK/MyProject/orcseepy/test/img/s3.png"
    main(file_path)

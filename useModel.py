from tensorflow import keras
import numpy as np
import cv2

# Параметры данных и модели
img_height, img_width = 32, 128
characters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
num_classes = len(characters) + 1

# Словари для преобразования символов в числа и обратно
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
num_to_char = {idx + 1: char for idx, char in enumerate(characters)}

# Загрузка модели
model = keras.models.load_model('crnn_text_recognition_model.h5')

def preprocess_image(image):
    # Изменение размера изображения до (32, 128)
    img = cv2.resize(image, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def get_text_bounding_boxes(predictions, original_width, original_height):
    predicted_probs = np.max(predictions, axis=-1)
    indices = range(predicted_probs.shape[1])  # Без порога, просто все индексы
    # Получаем координаты для каждого символа
    results = []
    for idx in indices:
        # Переводим индекс символа в координаты на оригинальном изображении
        x_start = int(idx * (original_width / predictions.shape[1]))
        x_end = int((idx + 1) * (original_width / predictions.shape[1]))  # Учитываем ширину символа
        # Преобразуем в проценты
        x_start_percent = (x_start / original_width) * 100
        x_end_percent = (x_end / original_width) * 100
        results.append((x_start_percent, x_end_percent))
    return results


def group_characters_to_words(predicted_labels, bounding_boxes):
    words = []
    current_word = []
    current_word_coords = []
    for i, num in enumerate(predicted_labels[0]):
        if num > 0:  # Если символ распознан
            char = num_to_char.get(num, '')
            current_word.append(char)
            # Преобразуем координаты символов в пиксели на оригинальном изображении
            current_word_coords.append(bounding_boxes[i])
        else:
            if current_word:  # Завершение текущего слова
                words.append({
                    "content": ''.join(current_word),
                    "coordinates": [current_word_coords[0], current_word_coords[-1]],  # Первые и последние координаты
                    "signature": False  # Установите свою логику для определения подписи
                })
                current_word = []
                current_word_coords = []
    # Добавляем последнее слово, если оно существует
    if current_word:
        words.append({
            "content": ''.join(current_word),
            "coordinates": [current_word_coords[0], current_word_coords[-1]],
            "signature": False
        })
    return words

def predict_image(image, original_width, original_height):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_labels = np.argmax(predictions, axis=-1)
    bounding_boxes = get_text_bounding_boxes(predictions, original_width, original_height)
    words_output = group_characters_to_words(predicted_labels, bounding_boxes)
    return words_output

# Загрузка изображения
image_path = 'examle1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Image at path {image_path} could not be loaded.")

original_height, original_width = image.shape
result_output = predict_image(image, original_width, original_height)
print(result_output)

# Вывод координат каждого слова
for word in result_output:
    print(f"Word: {word['content']}")
    print(f"  Coordinates: {word['coordinates']}")

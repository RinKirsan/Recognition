from tensorflow import keras
import numpy as np
import cv2
# Параметры данных и модели
img_height, img_width = 32, 128  # размеры изображений
characters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'  # кириллица и латиница
num_classes = len(characters) + 1  # количество классов + 1 для пустого символа
# Словари для преобразования символов в числа и обратно
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}  # +1 для смещения индекса
num_to_char = {idx + 1: char for idx, char in enumerate(characters)}
# Load the pre-trained model from an H5 file
model = keras.models.load_model('crnn_text_recognition_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img_height, img_width = 32, 128  # Define your image dimensions
    img = cv2.resize(image, (img_width, img_height))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return np.expand_dims(img, axis=0)  # Add batch dimension


def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)

    # Convert predictions to text using the reverse mapping
    predicted_labels = np.argmax(predictions, axis=-1)
    text_output = ''.join([num_to_char.get(num, '') for num in predicted_labels[0]])

    return text_output

# Load an image (example)
image_path = 'test3.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Make a prediction
result_text = predict_image(image)
print(f"Predicted Text: {result_text}")
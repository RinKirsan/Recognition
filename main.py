import cv2
import os
import numpy as np
import pandas as pd
import keras
import tensorflow.python.keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM
from keras import Model
from keras import backend as K
# Параметры данных и модели
img_height, img_width = 32, 128  # размеры изображений
characters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'  # кириллица и латиница
num_classes = len(characters) + 1  # количество классов + 1 для пустого символа
# Словари для преобразования символов в числа и обратно
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}  # +1 для смещения индекса
num_to_char = {idx + 1: char for idx, char in enumerate(characters)}

def create_crnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Сверточные слои
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Дополнительный MaxPooling для уменьшения высоты и ширины

    # Reshape to (timesteps, features)
    timesteps = 4  # This should be adjusted based on your architecture
    features = x.shape[1] * x.shape[2] * 128
    x = Reshape((32, 2 * 128))(x)

    # Рекуррентные слои
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)

    # Полносвязный слой
    outputs = Dense(num_classes, activation='softmax')(x)

    # Создание модели
    model = Model(inputs, outputs)
    return model

# Загрузка изображений и меток из TSV файла
base_path = "/home/user/Recognition/train/train"  # Путь к папке с изображениями
def load_data(data_tsv):
    try:
        data = pd.read_csv(data_tsv, sep='\t', header=0)  # Указываем, что первый ряд — заголовок
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return None, None

    images = []
    labels = []

    for idx, row in data.iterrows():
        try:
            # Формируем полный путь к изображению
            img_path = os.path.join(base_path, row['image_path'])
            label_text = row['label']

            # Проверка, существует ли файл по пути img_path
            if not os.path.exists(img_path):
                print(f"Файл не найден по пути: {img_path}")
                continue

            # Загрузка изображения
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Не удалось загрузить изображение по пути: {img_path}")
                continue  # Переходим к следующей строке

            # Преобразование текста в числовую последовательность
            label = [char_to_num[char] for char in label_text if char in char_to_num]
            labels.append(label)
            # Изменение размера и нормализация
            img = cv2.resize(img, (img_width, img_height))
            img = img / 255.0
            images.append(np.expand_dims(img, axis=-1))



        except KeyError as e:
            print(f"Ключ '{e}' не найден в строке {idx}. Проверьте имена столбцов.")
            continue
        except Exception as e:
            print(f"Неизвестная ошибка при обработке строки {idx}: {e}")
            print(label_text)
            continue

    images = np.array(images, dtype=np.float32)
    max_sequence_length = img_width // 4  # Change this to 32 if needed
    labels = keras.utils.pad_sequences(labels, maxlen=max_sequence_length, padding='post', value=0)

    return images, labels


# Загрузка данных
train_images, train_labels = load_data('/home/user/Recognition/train/train3.tsv')
val_images, val_labels = train_images, train_labels

# Преобразование меток в формат one-hot encoding
train_labels = keras.utils.to_categorical(train_labels, num_classes=num_classes)
val_labels = keras.utils.to_categorical(val_labels, num_classes=num_classes)

# Создание и компиляция модели
input_shape = (img_height, img_width, 1)
print("Train labels shape:", train_labels.shape)
print("Validation labels shape:", val_labels.shape)
model = create_crnn(input_shape, num_classes)

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Обучение модели
epochs = 50
batch_size = 32
print(len(train_images),len(train_labels))
print(train_labels[len(train_labels)//2])
print(len(val_images),len(val_labels))
print(val_labels[len(val_labels)//2])
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=epochs,
    batch_size=batch_size

)

# Сохранение модели
model.save('crnn_text_recognition_model.h5')

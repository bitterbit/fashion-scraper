import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as pltx

import os
import pathlib

CLASS_NAMES = ["positive", "negative"]
IMG_HEIGHT = IMG_WIDTH = 32 


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    print (parts[-2])
    return parts[-2]
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    print("processpath")
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label



def main():
    data_dir = pathlib.Path("traindata-small/train")
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    for image, label in labeled_ds.take(100):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())


    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    model.summary() 

    model.compile(optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    print(dir(labeled_ds))
    
    history = model.fit(labeled_ds, epochs=4) 

    print("history?", history)
    
if __name__ == '__main__':
    main()

import tensorflow as tf
import resizer
import sys
import cv2
import os.path
import numpy as np

import IPython.display as display
from PIL import Image

MODEL_PATH = "model.tf"
IMAGE_SIZE = 128

def load_image(path):
    img = cv2.imread(path)
    img = resizer.fill_to_square(img)
    img = resizer.resize(img, IMAGE_SIZE) 
    img = np.asarray(img) / (255.0) # normalize Data
    img = img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    return img

def predict(model, img):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(img)[0]
    # print("predictions!", predictions)
    if predictions[0] > predictions[1]:
        return "negative"
    else:
        return "positive"
    
                            
def main():
    if len(sys.argv) < 2:
        print ("not enough arguments")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    img_dir = sys.argv[1]

    for img_name in os.listdir(img_dir):
        if img_name.endswith(".jpg"):

            path = os.path.join(img_dir, img_name)
            img = load_image(path)
            result = predict(model, img)
            print("is person?:", result)
            Image.open(path).show()
            input("press anything to continue")

if __name__ == '__main__':
    main()

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import os
import pathlib
import cv2
import numpy as np

class ImageLoader(object):

    def __init__(self,PATH='', IMAGE_SIZE = 50):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        # This will get List of categories
        self.list_categories = []

    def get_categories(self):
        for path in os.listdir(self.PATH):
            if '_DS_Store' in path or '.DS_Store' in path:
                pass
            else:
                self.list_categories.append(path)
        return self.list_categories

    def Process_Image(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:        # if any image is corrupted
                        image_data_temp = cv2.imread(new_path)                 # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize,class_index])

                    except:
                        print ("error loading image: " + new_path)
                        pass

            data = np.asanyarray(self.image_data)
            # Iterate over the Data
            for x in data:
                self.x_data.append(x[0])        # Get the X_Data
                self.y_data.append(x[1])        # get the label

            X_Data = np.asarray(self.x_data) / (255.0)      # Normalize Data
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)

            return X_Data, Y_Data
        except:
            print("Failed to run Function Process Image ")

    def load_dataset(self):

        X_Data,Y_Data = self.Process_Image()
        return X_Data,Y_Data

def main():
    # Loading DATA
    train_path='traindata-small/train'
    test_path='traindata-small/test'
    train_data = ImageLoader(PATH=train_path, IMAGE_SIZE=128)
    test_data = ImageLoader(PATH=test_path, IMAGE_SIZE=128)
    train_data_x, train_data_y = train_data.load_dataset()
    test_data_x, test_data_y = test_data.load_dataset()

    print("LOAD DATA!")

    # Build Model
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2))
    model.summary() 

    model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    # Train
    print("TRAINNIG....")
    history = model.fit(train_data_x, train_data_y, epochs=10) 

    # Test
    print("TEST....")
    history = model.evaluate(test_data_x, test_data_y) 

if __name__ == '__main__':
    main()

import os
import cv2
import numpy as np

import resizer

class ImageLoader(object):

    def __init__(self,PATH='', IMAGE_SIZE = 50):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        self.blacklist = ['_DS_Store', '.DS_Store']

    def get_categories(self):
        categories = []
        for path in os.listdir(self.PATH):
            if self.is_valid_image_path(path):
                categories.append(path)

        return categories

    def is_valid_image_path(self, path):
        for keyword in self.blacklist:
            if keyword in path:
                return False

        return True

    def try_process_image(self):
        try:
            return self.process_image()
        except Exception as e:
            print("Failed to run function process image. error:", e)
        return None 

    def load_image(self, path):
        img = cv2.imread(path) # Read Image as numbers
        img_square = resizer.fill_to_square(img)
        img_resized = resizer.resize(img_square, self.IMAGE_SIZE) 
        return img_resized

    def try_load_image(self, path):
        if not self.is_valid_image_path(path):
            return None 

        # if any image is corrupted
        try:
            return self.load_image(path)
        except Exception as e:
            print ("error loading image:", path, "error: ", e)

    def process_image(self):
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
                new_img = self.try_load_image(new_path)
                if new_img is not None:
                    self.image_data.append([new_img, class_index])

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

    def load_dataset(self):
        X_Data,Y_Data = self.try_process_image()
        return X_Data,Y_Data


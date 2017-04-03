"""Generate data and train a classifier for OCR"""

import os
import numpy as np
import cv2
import sys

MIN_CONTOUR_AREA = 100
RESIZED_IMG_W = 20
RESIZED_IMG_H = 30
TRAININGFILE = "../../Images/TrainingCaractere/training_image_new2.jpg"

class Datagenerator:
    """Generator of data for training"""
    def __init__(self, trainingFile):
        """Init generator with training image"""

        self.training_file = trainingFile
        self.img_training = None

    def load_file(self):
        """Return if image training was loaded with success or not"""

        self.img_training = cv2.imread(self.training_file)
        return self.img_training is not None


    def generate(self):
        """Generate file for training"""

        grayimg = cv2.cvtColor(self.img_training, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(grayimg, (5, 5), 0)
        img_thresh = cv2.adaptiveThreshold(img_blurred, 255, 
                                           cv2.ADAPTIVE_THREST_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 
                                           11, 
                                           2)




if __name__ == "__main__":
    dg = Datagenerator(TRAININGFILE)

    if dg.load_file():
        print("Training image load with success")

    else:
        print("Error: The training image was not loaded")

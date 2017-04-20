"""Generate data and train a classifier for OCR"""

import os
import sys
import cv2
import numpy as np


MIN_CONTOUR_AREA = 1
RESIZED_IMG_W = 20
RESIZED_IMG_H = 30
TRAININGFILE = "../../Images/TrainingCaractere/a_b.jpg"

class WindowManager:
    """Manage les fenetres de l'application"""
    def __init__(self, name):
        """Initialisation du manager avec un nom"""
        self.name = name

    def generate_window(self, img):
        """Genere la fenetre et la gere"""
        cv2.imshow(self.name, img)



class Datagenerator:
    """Generator of data for training"""
    def __init__(self, trainingFile):
        """Init generator with training image"""
        self.training_file = trainingFile
        self.img_training = None
        self.w_m = WindowManager("Trainer")

    def load_file(self):
        """Return if image training was loaded with success or not"""
        self.img_training = cv2.imread(self.training_file)
        self.w_m.generate_window(self.img_training)
        return self.img_training is not None


    def generate(self):
        """Generate file for training"""
        grayimg = cv2.cvtColor(self.img_training, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(grayimg, (5, 5), 0)
        img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           11,
                                           2)
        self.w_m.generate_window(img_thresh)
        img_thresh_copy = img_thresh.copy()
        img_contour, npa_contours, npa_hierarchy = cv2.findContours(img_thresh_copy,
                                                                    cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
        npa_flattened_images = np.empty((0, RESIZED_IMG_W * RESIZED_IMG_H))
        int_classifications = []
        int_valid_chars = [ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'),
                           ord('h'), ord('i'), ord('j'), ord('k'), ord('l'), ord('m'), ord('n'),
                           ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'), ord('u'),
                           ord('v'), ord('w'), ord('x'), ord('y'), ord('z'), ord('A'), ord('B'),
                           ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'),
                           ord('J'), ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'),
                           ord('Q'), ord('R'), ord('S'), ord('T'), ord('U'), ord('V'), ord('W'),
                           ord('X'), ord('Y'), ord('Z')]

        for npa_contour in npa_contours:
            if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:
                [int_x, int_y, int_w, int_h] = cv2.boundingRect(npa_contour)

                cv2.rectangle(self.img_training,
                              (int_x, int_y),
                              (int_x + int_w, int_y + int_h),
                              (0, 0, 255),
                              2)

                img_roi = img_thresh[int_y:int_y+int_h, int_x:int_x+int_w]
                img_roi_resized = cv2.resize(img_roi, (RESIZED_IMG_W, RESIZED_IMG_H))

                self.w_m.generate_window(img_roi)
                self.w_m.generate_window(img_roi_resized)
                self.w_m.generate_window(self.img_training)

                int_char = cv2.waitKey(0)
                if int_char == 27:
                    sys.exit()
                elif int_char in int_valid_chars:
                    int_classifications.append(int_char)
                    npa_flattened_image = img_roi_resized.reshape((1, RESIZED_IMG_W*RESIZED_IMG_H))
                    npa_flattened_images = np.append(npa_flattened_images, npa_flattened_image, 0)

        flt_classifications = np.array(int_classifications, np.float32)
        npa_classifications = flt_classifications.reshape((flt_classifications.size, 1))
        print("Training complete !!")

        np.savetxt("classification.txt", npa_classifications)
        np.savetxt("flattened.txt", npa_flattened_images)
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    DG = Datagenerator(TRAININGFILE)

    if DG.load_file():
        print("Training image load with success")
        DG.generate()
        exit()
    else:
        print("Error: The training image was not loaded")
        exit()

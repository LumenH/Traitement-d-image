import cv2
import numpy
import os

import CharDetector
from PanelDetector import PanelDetector

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

PANEL_WIDTH_PADDING_FACTOR = 1.3
PANEL_HEIGHT_PADDING_FACTOR = 1.5

showSteps = True

if __name__ == "__main__":
    KNNSuccess = CharDetector.trainKNN()

    if not KNNSuccess:
        print("Erreur pendant l'entrainement de KNN")
        exit()

    print("Chargement de l'image original")
    imgOriginal = cv2.imread("1.jpg")
    imgOriginal = cv2.resize(imgOriginal, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

    if imgOriginal is None:
        print("Image = None")
        os.system("pause")
        exit()

    print("Detection du panneau")
    panelDetector = PanelDetector(True, PANEL_WIDTH_PADDING_FACTOR, PANEL_HEIGHT_PADDING_FACTOR)
    listPanneauPossible = panelDetector.detectPanelInScene(imgOriginal)

    print(listPanneauPossible)

    print("Detection de text à l'intérieur des panneaux")
    #listTextePossible = Detect



import cv2
import numpy as np
import math

class CaracterePossible:
    def __init__(self, contour):
        self.contour = contour

        self.boundingRect = cv2.boundingRect(self.contour)
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.boundingX = intX
        self.boundingY = intY
        self.boundingLargeur = intWidth
        self.boundingHauteur = intHeight
        self.boundingArea = self.boundingLargeur*self.boundingHauteur
        self.centerX = (2*self.boundingX + self.boundingLargeur)/2
        self.centerY = (2*self.boundingY + self.boundingHauteur)/2
        self.diagonalSize = math.sqrt((self.boundingLargeur**2)+(self.boundingHauteur**2))
        self.RatioAspect = float(self.boundingLargeur)/float(self.boundingHauteur)


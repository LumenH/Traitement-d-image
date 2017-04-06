"""Module charger de reconnaitre les caractères inscrit sur les panneaux routiers"""
import cv2
import math



'''Class for the characters'''
class PossibleChar:

    def __init__(self, _contour):
        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))
        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(slef.intBoundingRectHeight)

'''Class for the plate'''
class PossiblePlate:

    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None #sans doute pas besoin, puisque c'est les niveaux de gris
        self.imgThresh = None
        self.rrLocationOfPlateInScene = None
        self.strChars=""

#Main method
if __name__ == "__main__":

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import PanneauPossible
import CharDetector
from CaracterePossible import CaracterePossible

showStep = True

#PLATE_WIDTH_PADDING_FACTOR = WPF
#PLATE_HEIGHT_PADDING_FACTOR = HPF

class PanelDetector:
    def __init__(self, showStep, WPF, HPF):
        print("Construction du detecteur de panneau")
        self.showStep = showStep
        self.WPF = WPF
        self.HPF = HPF

    def showState(self, img, title):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findChar(self, imgThresh):
        print("Recherche caractere")
        listCaracterePossible = []
        compteurCaracterePossible = 0
        imgThreshCopy = imgThresh.copy()
        imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        hauteur, largeur = imgThresh.shape
        imgContours = np.zeros((hauteur, largeur, 3), np.uint8)
        print("Check si les contours trouvé sont de possibles caracteres")
        for i in range(0, len(contours)):
            if self.showStep:
                cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
            caracterePossible = CaracterePossible(contours[i])
            if CharDetector.check(caracterePossible):
                compteurCaracterePossible = compteurCaracterePossible+1
                listCaracterePossible.append(caracterePossible)
        if self.showStep:
            self.showState(imgContours, "Contours")
        return listCaracterePossible

    def detectPanelInScene(self, imgOriginal):
        print("Detection des panneaux dans la scene")
        listPanneauPossible = []
        hauteur, largeur, nombreChan = imgOriginal.shape
        imgSceneGrayScale = np.zeros((hauteur, largeur, 1), np.uint8)
        imgSceneThreshold = np.zeros((hauteur, largeur, 1), np.uint8)
        imgContours = np.zeros((hauteur, largeur, 3), np.uint8)
        cv2.destroyAllWindows()
        if self.showStep:
            self.showState(imgOriginal, "Image original")
        imgSceneGrayScale, imgSceneThreshold = Preprocess.run(imgOriginal)
        if self.showStep:
            self.showState(imgSceneGrayScale, "Image en niveau de gris")
            self.showState(imgSceneThreshold, "Image en threshold")
        listCaracterePossible = self.findChar(imgSceneThreshold)
        if self.showStep:
            imgContours = np.zeros((hauteur, largeur, 3), np.uint8)
            contours= []
            for caracterePossible in listCaracterePossible:
                contours.append(caracterePossible.contour)
            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            self.showState(imgContours, "Contours des caracteres possible")

        print("Trouver les listes de caractere qui match")
        listMatchingCaractere = CharDetector.findMatching(listCaracterePossible)
        if self.showStep:
            imgContours = np.zeros((hauteur, largeur, 3), np.uint8)
            for listm in listMatchingCaractere:
                blue = random.randint(0,255)
                green = random.randint(0, 255)
                red = random.randint(0, 255)
                contours = []
                for matchCaractere in listm:
                    contours.append(matchCaractere.contour)
                cv2.drawContours(imgContours, contours, -1, (blue, green, red))
            self.showState(imgContours, "MatchingContours")
        for listMatchingCaractere in listMatchingCaractere:
            panneauPossible = self.extractPanel(imgOriginal, listMatchingCaractere)
            if panneauPossible.imgPanel is not None: 
                listPanneauPossible.append(panneauPossible)
        print("\n" + str(len(listPanneauPossible)) + "panneau possible trouvé")
        if self.showStep:
            self.showState(imgContours, "Contours")
            for i in range(0, len(listPanneauPossible)):
                s = cv2.boxPoints(listPanneauPossible[i].location)
                cv2.line(imgContours, tuple(s[0]), tuple(s[1]), Main.SCALAR_RED, 2)
                cv2.line(imgContours, tuple(s[1]), tuple(s[2]), Main.SCALAR_RED, 2)
                cv2.line(imgContours, tuple(s[2]), tuple(s[3]), Main.SCALAR_RED, 2)
                cv2.line(imgContours, tuple(s[3]), tuple(s[0]), Main.SCALAR_RED, 2)
                self.showState(imgContours, "Box points")
        return listPanneauPossible

    def extractPanel(self, imgOriginal, listOfMatchingChars):
        possiblePanel = PanneauPossible.PanneauPossible()
        listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.centerX)
        fltPanelCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX)/2.0
        fltPanelCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY)/2.0
        ptPanelCenter = fltPanelCenterX, fltPanelCenterY
        panelLargeur = int((listOfMatchingChars[len(listOfMatchingChars)-1].boundingX + listOfMatchingChars[len(listOfMatchingChars) - 1].boundingLargeur - listOfMatchingChars[0].boundingX)* self.WPF)
        totalOfCharsHauteur = 0
        for matchingChar in listOfMatchingChars:
            totalOfCharsHauteur = totalOfCharsHauteur + matchingChar.boundingHauteur
        moyenneHauteur = totalOfCharsHauteur / len(listOfMatchingChars)
        panelHauteur = int(moyenneHauteur * self.HPF)
        opposite = listOfMatchingChars[len(listOfMatchingChars) -1].centerY - listOfMatchingChars[0].centerY
        hypo = CharDetector.distanceEntreChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars)-1])
        correctionAngle = math.asin(opposite/hypo)
        correctionAngleDeg = correctionAngle * (180.0/math.pi)
        PanneauPossible.location = (tuple(ptPanelCenter), (panelLargeur, panelHauteur), correctionAngleDeg, 1.0)
        rot = cv2.getRotationMatrix2D(tuple(ptPanelCenter), correctionAngleDeg, 1.0)
        hauteur, largeur, numChan = imgOriginal.shape
        imgRot = cv2.warpAffine(imgOriginal, rot, (largeur, hauteur))
        imgCropped = cv2.getRectSubPix(imgRot, (panelLargeur, panelHauteur), tuple(ptPanelCenter))
        PanneauPossible.imgPanel = imgCropped
        return possiblePanel


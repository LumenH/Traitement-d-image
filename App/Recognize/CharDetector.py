import cv2
import numpy as np
import math
import random
import os

import Main

kNearest = cv2.ml.KNearest_create()

#Variable pour savoir si un caractere est possible
MIN_PIXEL_WIDTH = 10
MIN_PIXEL_HEIGHT = 50
MIN_ASPECT_RATIO = 0.5 #evt 0.25
MAX_ASPECT_RATIO = 2.5
MIN_PIXEL_AREA = 1000
MAX_AREA = 80000


#A optimiser demain !!!!! 
MIN_DIAG_SIZE_MULTIPLE_AWAY = 1000
MAX_DIAG_SIZE_MULTIPLE_AWAY = 2000
MAX_CHANGE_IN_AREA = 5000
MAX_CHANGE_IN_WIDTH = 8000
MAX_CHANGE_IN_HEIGHT = 2000
MAX_ANGLE_BETWEEN_CHARS = 4005.0
MIN_NUMBER_OF_MATCHING_CHARS = 1

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100

def trainKNN():
    print("Entrainement du KNN")
    allContoursWithData = []
    validContourWithData = []
    print("Train KNN")
    try:
        npaClassifications = np.loadtxt("classification.txt", np.float32)
    except:
        print("Error lors du chargement de classification.txt")
        os.system("pause")
        return False
    try:
        npaFlattenedImages = np.loadtxt("flattened.txt", np.float32)
    except:
        print("Erreur lors du chargement des images aplaties")
        os.system("pause")
        return False
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest.setDefaultK(1)
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    return True

def check(caracterePossible):
    if caracterePossible.boundingArea > MIN_PIXEL_AREA and caracterePossible.boundingLargeur > MIN_PIXEL_WIDTH and caracterePossible.boundingHauteur > MIN_PIXEL_HEIGHT and MIN_ASPECT_RATIO < caracterePossible.RatioAspect and caracterePossible.RatioAspect < MAX_ASPECT_RATIO and caracterePossible.boundingArea < MAX_AREA:
        return True
    else:
        return False

def findMatching(listCaracterePossible):
    listOflistMatchingChar = []
    for caracterePossible in listCaracterePossible:
        listofMatchingChar = findingListMatching(caracterePossible, listCaracterePossible)
        print("Longueur de matchingChar" + str(len(listofMatchingChar)))
        listofMatchingChar.append(caracterePossible)
        if len(listofMatchingChar) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue
        listOflistMatchingChar.append(listofMatchingChar)
        listCaracterePossibleMatchRemoved = []
        listCaracterePossibleMatchRemoved = list(set(listCaracterePossible)- set(listofMatchingChar))
        recursiveListOfList = findMatching(listCaracterePossibleMatchRemoved)
        for recursive in recursiveListOfList:
            listOflistMatchingChar.append(recursive)
        break
    return listOflistMatchingChar

def findingListMatching(caracterePossible, listCaractere):
    listMatch = []
    for matchingPossible in listCaractere:
        if matchingPossible == caracterePossible:
            continue
        fltdistanceEntreCaractere = distanceEntreChars(caracterePossible, matchingPossible)
        fltAngle = anglEntreCaractere(caracterePossible, matchingPossible)
        fltChangeArea = float(abs(matchingPossible.boundingArea - caracterePossible.boundingArea))/float(caracterePossible.boundingArea)
        fltChangeLargeur = float(abs(matchingPossible.boundingLargeur - caracterePossible.boundingLargeur))/float(caracterePossible.boundingLargeur)
        fltChangeHauteur = float(abs(matchingPossible.boundingHauteur - caracterePossible.boundingHauteur))/float(caracterePossible.boundingHauteur)
        if fltdistanceEntreCaractere < (caracterePossible.diagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and fltAngle < MAX_ANGLE_BETWEEN_CHARS and fltChangeArea < MAX_CHANGE_IN_AREA and fltChangeHauteur < MAX_CHANGE_IN_HEIGHT and fltChangeLargeur < MAX_CHANGE_IN_WIDTH:
            listMatch.append(matchingPossible)
        return listMatch


def anglEntreCaractere(first, second):
    adj = float(abs(first.centerX - second.centerX))
    opp = float(abs(first.centerY - second.centerY))
    if adj != 0.0:
        angle = math.atan(opp/adj)
    else:
        angle = 1.57
    angle = angle*(180.0/math.pi)
    return angle


def distanceEntreChars(first, second):
    x = abs(first.centerX - second.centerX)
    y = abs(first.centerY - second.centerY)
    return math.sqrt((x**2)+(y**2))



def detectCharsinPanel(listPanneauPossible):
    PanelCounter = 0
    imgContours = None
    contours = []
    if len(listPanneauPossible) <= 0:
        print("Pas de panneau sur lesquels detecter des caractères")
        return listPanneauPossible

#    for panneauPossible in listPanneauPossible:
#        panneauPossible.imgGrayScale, panneauPossible.imgTresh = Preprocess.run(panneauPossible.imgPanel)

#todo
#detectCharsInPlates
#findPossibleCharsInPlate
#def removeInnerOverlappingChars(listOfMatchingChars):
#recognizeCharsInPlate
#finir le main





import cv2
import numpy as np
import os

import CharDetector
import PlateDetector
import PossiblePlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main():
    blnKNNTrainingSuccessful = CharDetector.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nError : KNN training was not successful")
        return

    imgOriginalScene = cv2.imread()#Ouvrir l'image à tester

    if imgOriginalScene is None:
        print("\nError : image notre read from file")
        os.system("pause")
        return

    listOfPossiblePlates = PlateDetector.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = PlateDetector.detectCharsinPlate(listOfPossiblePlates)

    cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        print("\nNo plate detected")
    else:
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        cityPlate = listOfPossiblePlates[0]
        cv2.imshow("imgPlate", cityPlate.imgPlate)
        cv2.imshow("imgThresh", cityPlate.imgThresh)

        if len(cityPlate.strChars) == 0:
            print("\nNor characters detected")
            return

        drawRedRectangleAroundPlate(imgOriginalScene, cityPlate)
        print("\nCity plate read from image = " +cityPlate.strChars)

        writeCityPlateCharsOnImage(imgOriginalScene, cityPlate)

        cv2.imshow("imgOriginalScene", imgOriginalScene)
        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    cv2.waitKey(0)
    return


def drawRedRectangleAroundPlate(imgOriginalScene, cityPlate):
    p2fRectPoints = cv2.boxPoints(cityPlate.rrLocationOfPlateInScene)

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeCityPlateCharsOnImage(imgOriginalScene, cityPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = cityPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) /30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, basline = cv2.getTextSize(cityPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg) = cityPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)
    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth/2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight/2))

    cv2.putText(imgOriginalScene, cityPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)


if __name__ == "__main__":
    main()

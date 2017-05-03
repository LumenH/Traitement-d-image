import cv2
import numpy as np

class PanneauPossible:
    def __init__(self):
        self.imgPanel = None
        self.imgGray = None
        self.imgThresh = None
        self.location = None
        self.str = ""
from checkGBFace_solvePnP import isGoodFace_solvePnp
from checkGBFace_landmarkDistance import isGoodFace_landmarkDistanceBased
from checkBlurFace import isBlurFace
from imutils import face_utils

import dlib
import cv2
import numpy as np


class GoodFaceChecker(object):
    def __init__(self, method='solvePnP', checkBlur=False):
        self.method = method
        self.checkBlur = checkBlur

    def isGoodFace(self, face_image, original_frame_size, landmark_points=None):
        if self.checkBlur and isBlurFace(face_image):
            return False

        if landmark_points is None:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            landmark_predictor = dlib.shape_predictor(
                "../models/dlib_facial_landmark_detector/shape_predictor_68_face_landmarks.dat")
            rect = dlib.rectangle(0, 0, face_image.shape[0], face_image.shape[1])
            shape = landmark_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            landmark_points = np.array([shape[38], shape[46],
                               shape[30], shape[60], shape[54]], dtype='double')

        if self.method == 'solvePnP':
            return isGoodFace_solvePnp(landmark_points, original_frame_size)
        elif self.method == 'landmarkDistance':
            return isGoodFace_landmarkDistanceBased(landmark_points)

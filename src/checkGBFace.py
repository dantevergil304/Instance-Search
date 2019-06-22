from checkGBFace_solvePnP import isGoodFace_solvePnp
from checkGBFace_landmarkDistance import isGoodFace_landmarkDistanceBased
from checkGBFace_classifier import isGoodFace_classifier
from checkBlurFace import isBlurFace
from imutils import face_utils

# import dlib
import cv2
import numpy as np
import json


class GoodFaceChecker(object):
    def __init__(self, method='landmark_based', checkBlur=False):
        self.method = method  # Main Method
        self.checkBlur = checkBlur

    def isGoodFace(self, face_image, original_frame_size=None, landmark_points=None, classifier_type=None):
        if self.checkBlur and isBlurFace(face_image):
            return False

        ####### UNCOMMENT WHEN USING ON gpu6 and gpu3 #######
        # if landmark_points is None:
        #     gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        #     landmark_predictor = dlib.shape_predictor(
        #         "../models/dlib_facial_landmark_detector/shape_predictor_68_face_landmarks.dat")
        #     rect = dlib.rectangle(
        #         0, 0, face_image.shape[0], face_image.shape[1])
        #     shape = landmark_predictor(gray, rect)
        #     shape = face_utils.shape_to_np(shape)
        #     landmark_points = np.array([shape[38], shape[46],
        #                                 shape[30], shape[60], shape[54]], dtype='double')
        ####### UNCOMMENT WHEN USING ON gpu6 and gpu3 #######

        # for lm in landmark_points:
        #     cv2.circle(face_image, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), 2)

        # cv2.imshow('face', face_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        with open('../cfg/search_config.json', 'r') as f:
            cfg = json.load(f)
        if self.method == 'landmark_based':
            checkFacePoseMethod = cfg['rmBadFacesLandmarkBasedParams']['check_face_pose_method']
            if checkFacePoseMethod == 'solvePnP':
                return isGoodFace_solvePnp(landmark_points, original_frame_size)
            elif checkFacePoseMethod == 'landmarkDistance':
                return isGoodFace_landmarkDistanceBased(landmark_points)
        elif self.method == 'classifier':
            return isGoodFace_classifier(face_image, classifier_type)

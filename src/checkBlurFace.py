import numpy as np
import cv2


def isBlurFace(img):
    MIN_SHARPEN_THRESHOLD = 0.3
    MIN_SHARPEN_VARIANCE = 0.002

    height, width, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    result1 = np.sum(laplacian[0:int(height/2), 0:int(width/4)])

    result2 = np.sum(
        laplacian[0:int(height/2), int(width/2):int(width/4*3)])

    result3 = np.sum(laplacian[int(height/2):height, 0:int(width/4)])

    result4 = np.sum(
        laplacian[int(height/2):height, int(width/2):int(width/4*3)])

    variance = (cv2.Laplacian(gray, cv2.CV_64F).var())/(width*height)

    if round(result1/(height*width/4), 3) > MIN_SHARPEN_THRESHOLD \
            and round(result2/(height*width/4), 3) > MIN_SHARPEN_THRESHOLD \
            and round(result3/(height*width/4), 3) > MIN_SHARPEN_THRESHOLD \
            and round(result4/(height*width/4), 3) > MIN_SHARPEN_THRESHOLD \
            and variance > MIN_SHARPEN_VARIANCE:
        return False

    return True

import math


def isGoodFace_landmarkDistanceBased(landmark_points):
    MAX_DIFFERENCE_DISTANCE_LEFTEYE_NOSE_RIGHTEYE_NOSE = 20

    left_eye = landmark_points[0]
    right_eye = landmark_points[1]
    nose = landmark_points[2]
    left_mouth_corner = landmark_points[3]
    right_mouth_corner = landmark_points[4]

    distance_lefteye_nose = math.sqrt(
        (int(left_eye[0])-int(nose[0]))**2 + (int(left_eye[1])-int(nose[1]))**2)
    distance_righteye_nose = math.sqrt(
        (int(right_eye[0])-int(nose[0]))**2 + (int(right_eye[1])-int(nose[1]))**2)

    # check if nose landmark point is on one side (i.e. not in the middle of too eyes or two mouth corners)
    if int(nose[0]) < int(left_eye[0]) \
            or int(nose[0]) > int(right_eye[0]) \
            or int(nose[0]) < int(left_mouth_corner[0]) \
            or int(nose[0]) > int(right_mouth_corner[0]):
        return False

    # check if distance of each eye to nose is fairly equal
    if abs(distance_lefteye_nose-distance_righteye_nose) >= \
            MAX_DIFFERENCE_DISTANCE_LEFTEYE_NOSE_RIGHTEYE_NOSE:
        return False

    return True

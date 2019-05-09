import cv2
import numpy as np
import re
import math
import glob
import os
import pickle


def get3dModelPoints():
    # Refer to line 69 in glAnthropometric3DModel.cpp by pedromartins
    # https://github.com/jensgrubert/glAnthropometric3DModel/blob/master/glAnthropometric3DModel.cpp
    pts_str = '{-7.308957,0.913869,0.000000}, {-6.775290,-0.730814,-0.012799}, {-5.665918,-3.286078,1.022951}, {-5.011779,-4.876396,1.047961}, {-4.056931,-5.947019,1.636229}, {-1.833492,-7.056977,4.061275}, {0.000000,-7.415691,4.070434}, {1.833492,-7.056977,4.061275}, {4.056931,-5.947019,1.636229}, {5.011779,-4.876396,1.047961}, {5.665918,-3.286078,1.022951}, {6.775290,-0.730814,-0.012799}, {7.308957,0.913869,0.000000}, {5.311432,5.485328,3.987654}, {4.461908,6.189018,5.594410}, {3.550622,6.185143,5.712299}, {2.542231,5.862829,4.687939}, {1.789930,5.393625,4.413414}, {2.693583,5.018237,5.072837}, {3.530191,4.981603,4.937805}, {4.490323,5.186498,4.694397}, {-5.311432,5.485328,3.987654}, {-4.461908,6.189018,5.594410}, {-3.550622,6.185143,5.712299}, {-2.542231,5.862829,4.687939}, {-1.789930,5.393625,4.413414}, {-2.693583,5.018237,5.072837}, {-3.530191,4.981603,4.937805}, {-4.490323,5.186498,4.694397}, {1.330353,7.122144,6.903745}, {2.533424,7.878085,7.451034}, {4.861131,7.878672,6.601275}, {6.137002,7.271266,5.200823}, {6.825897,6.760612,4.402142}, {-1.330353,7.122144,6.903745}, {-2.533424,7.878085,7.451034}, {-4.861131,7.878672,6.601275}, {-6.137002,7.271266,5.200823}, {-6.825897,6.760612,4.402142}, {-2.774015,-2.080775,5.048531}, {-0.509714,-1.571179,6.566167}, {0.000000,-1.646444,6.704956}, {0.509714,-1.571179,6.566167}, {2.774015,-2.080775,5.048531}, {0.589441,-2.958597,6.109526}, {0.000000,-3.116408,6.097667}, {-0.589441,-2.958597,6.109526}, {-0.981972,4.554081,6.301271}, {-0.973987,1.916389,7.654050}, {-2.005628,1.409845,6.165652}, {-1.930245,0.424351,5.914376}, {-0.746313,0.348381,6.263227}, {0.000000,0.000000,6.763430}, {0.746313,0.348381,6.263227}, {1.930245,0.424351,5.914376}, {2.005628,1.409845,6.165652}, {0.973987,1.916389,7.654050}, {0.981972,4.554081,6.301271}'

    pts = pts_str.split(', ')

    ret = []
    for pt in pts:
        x, y, z = [float(coord)
                   for coord in re.findall(r'-*[0-9]+.[0-9]+', pt)]
        ret.append((x, y, z))
    return ret


def getFaceRotationMatrix(image_points, image_size):
    # 3d model points
    # pts=get3dModelPoints()
    # model_points=np.array([
    #     # Left eye
    #     (pts[15][0], (pts[15][1]+pts[19][1])/2, -(pts[15][2]+pts[19][2])/2),
    #     # Right eye
    #     (pts[23][0], (pts[23][1]+pts[27][1])/2, -(pts[23][2]+pts[27][2])/2),
    #     # Nose tip
    #     pts[52],
    #     # Left mouth corner
    #     pts[43],
    #     # Right mouth corner
    #     pts[39]
    # ])
    # model_points = np.array([
    #     (-175.0, 170.0, -130.0),   # Left eye
    #     (175.0, 170.0, -130.0),    # Right eye
    #     (0.0, 0.0, 0.0),           # Nose tip
    #     (-150.0, -150.0, -125.0),  # Left mouth corner
    #     (150.0, -150.0, -125.0)    # Right mouth corner
    # ])
    model_points = np.array([
        (-36.9522, 39.3518, 47.1217),
        (35.446, 38.4345, 47.6468),
        (-0.0697709, 18.6015, 87.9695),
        (-27.6439, -29.6388, 73.8551),
        (28.7793, -29.2935, 72.7329),
    ])

    # camera intrinsic parameters
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double'
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points,
                                                          image_points, camera_matrix,
                                                          dist_coeffs,
                                                          flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix, translation_vector


def getFaceRotationAngles(image_points, image_size):
    # return rmat2agl(getFaceRotationMatrix(image_points, image_size)[0])
    rmat, tvec = getFaceRotationMatrix(image_points, image_size)
    projmat = np.concatenate((rmat, tvec), axis=1)
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(projmat)
    return np.squeeze(eulerAngles)


def isGoodFace_solvePnp(landmark_points, image_size, thresh=55):
    eulerAngles = getFaceRotationAngles(landmark_points, image_size)
    if abs(eulerAngles[1]) < thresh:
        return True
    return False


if __name__ == '__main__':
    for topic_dir in glob.glob('../topics_data/origin-queries/*'):
        topic_name = os.path.basename(topic_dir)
        for img_path in glob.glob(os.path.join(topic_dir, '*png')):
            img_name = os.path.basename(img_path)

            img_shape = cv2.imread(os.path.join(
                '../topics_data/topics-frames/', topic_name, img_name)).shape
            face_img = cv2.imread(img_path)

            with open(os.path.join(topic_dir, os.path.splitext(img_name)[0] + '_bb_landmark.pkl'), 'rb') as f:
                bbs, landmarks = pickle.load(f)

            image_points = []

            for i in range(int(len(landmarks)/2.)):
                x, y = int(landmarks[i]), int(landmarks[i+5])
                image_points.append((x, y))

                # draw landmark
                cv2.circle(
                    face_img, (x - bbs[0], y - bbs[1]), 2, (0, 255, 0), 2)

            image_points = np.array(image_points, dtype='double')

            eulerAngles = getFaceRotationAngles(image_points, img_shape)

            print('[+] Face Rotation Euler angles')
            print(eulerAngles)

            save_path = '../topics_data/processed-queries/solvePnP'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if abs(eulerAngles[1]) < 55:
                goodFaces_save_path = os.path.join(save_path, 'goodFaces')
                os.makedirs(goodFaces_save_path, exist_ok=True)
                cv2.imwrite(os.path.join(goodFaces_save_path, os.path.splitext(
                    img_name)[0] + f'_eulerAngles={eulerAngles}.png'), face_img)
            else:
                badFaces_save_path = os.path.join(save_path, 'badFaces')
                os.makedirs(badFaces_save_path, exist_ok=True)
                cv2.imwrite(os.path.join(badFaces_save_path, os.path.splitext(
                    img_name)[0] + f'_eulerAngles={eulerAngles}.png'), face_img)

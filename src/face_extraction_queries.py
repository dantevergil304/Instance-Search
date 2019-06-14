import sys
import numpy as np
import cv2
import json
import os
import glob
import tensorflow as tf

sys.path.append('../3rd_party/')
from ServiceMTCNN import detect_face


def detect_face_by_image(query, masks):
    '''
    Parameters:
    - query: list of query images
    - masks: list of mask images for each query
    Returns:
    - faces: list of detected faces in query. For each image in query,
    we just take the best face. In case no face was detected, skip.
    - bbs_coord: list of bbox coordinate of the return faces.
    - landmarks_coord: list of landmarks coordinate of the return faces.
    '''
    # TF session
    sess = tf.Session()
    # Create MTCNN
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    # MTCNN Hyperparameters
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    faces = []
    bbs_coord = []
    landmarks_coord = []
    for image, mask in zip(query, masks):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width = image.shape[0], image.shape[1]

        # detected faces
        detected_faces, detected_landmark = detect_face.detect_face(
            image, minsize, pnet, rnet, onet, threshold, factor)
        best_bbox = None
        best_landmark = None
        best_overlap = 0
        best_score = 0
        highest_y = float('inf')
        for face_info, landmark in zip(detected_faces, detected_landmark.transpose()):
            x, y, _x, _y = [int(coord) for coord in face_info[:4]]
            bbox_mask = np.zeros((im_height, im_width), dtype=np.uint8)
            cv2.rectangle(bbox_mask, (x, y), (_x, _y), 255, cv2.FILLED)
            ones_mask = np.ones((im_height, im_width), dtype=np.uint8)
            res = cv2.bitwise_and(mask, bbox_mask, mask=ones_mask)

            # Compute the overlapping area
            bbox_px_count = np.count_nonzero(bbox_mask)
            overlap = np.count_nonzero(res)

            # if overlap > 0 and y < highest_y:
            #     best_bbox = face_info
            #     best_landmark = landmark
            #     best_overlap = overlap
            #     highest_y = y

            if face_info[4] > 0.8:
                if overlap / bbox_px_count >= 0.5 and overlap > best_overlap:
                    best_bbox = face_info
                    best_landmark = landmark
                    best_overlap = overlap
            # if overlap > 0:
            #     if face_info[4] > best_score:
            #         best_bbox = face_info
            #         best_landmark = landmark
            #         best_score = face_info[4]

        # Result image after applying mask
        if best_bbox is not None:
            x, y, _x, _y = [int(coord) for coord in best_bbox[:4]]
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if _x >= im_width:
                _x = im_width
            if _y >= im_height:
                _y = im_height

            face = image[y: _y, x: _x]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            faces.append(face)
            bbs_coord.append((x, y, _x, _y))
            landmarks_coord.append(best_landmark)
        else:
            faces.append(None)
            bbs_coord.append(None)
            landmarks_coord.append(None)

    return faces, bbs_coord, landmarks_coord


def detect_face_by_path(query_path, masks_path):
    '''
    Parameters:
    - query_path: path to query
    - masks_path: path to masks
    Returns:
    - ret: list of detected faces in query. For each image in query,
    we just take the best face. In case no face was detected, skip.
    - query_path, masks_path: same as parameters
    '''
    query = [cv2.imread(im_path) for im_path in query_path]
    masks = [cv2.imread(mask_path, 0) for mask_path in masks_path]
    faces, bbs_coord, landmarks_coord = detect_face_by_image(query, masks)
    return zip(faces, query_path, masks_path), bbs_coord, landmarks_coord


if __name__ == "__main__":
    with open('../cfg/config.json', 'r') as f:
        config = json.load(f)

    names = ['archie', 'billy', 'ian', 'janine',
             'peggy', 'phil', 'ryan', 'shirley']
    for name in names:
        queries_folder = os.path.abspath(config['raw_data']['queries_folder'])
        query_path = [name + '.1.src.bmp', name + '.2.src.bmp',
                      name + '.3.src.bmp', name + '.4.src.bmp']
        query_path = [os.path.join(queries_folder, pth) for pth in query_path]
        masks_path = [name + '.1.mask.bmp', name + '.2.mask.bmp',
                      name + '.3.mask.bmp', name + '.4.mask.bmp']
        masks_path = [os.path.join(queries_folder, pth) for pth in masks_path]

        detect_face_by_path(query_path, masks_path)

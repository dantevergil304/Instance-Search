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
    - ret: list of detected faces in query. For each image in query,
    we just take the best face. In case no face was detected, skip.
    - result: list of resulting images after applying mask, having the
    same size as ret.
    '''
    # TF session
    sess = tf.Session()
    # Create MTCNN
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    # MTCNN Hyperparameters
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    ret = []
    bbs_coord = []
    # result = []
    for image, mask in zip(query, masks):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width = image.shape[0], image.shape[1]

        # detected faces
        detected_faces, _ = detect_face.detect_face(
            image, minsize, pnet, rnet, onet, threshold, factor)
        best_bbox = None
        best_overlap = 0
        for face_info in detected_faces:
            x, y, _x, _y = [int(coord) for coord in face_info[:4]]
            bbox_mask = np.zeros((im_height, im_width), dtype=np.uint8)
            cv2.rectangle(bbox_mask, (x, y), (_x, _y), 255, cv2.FILLED)
            ones_mask = np.ones((im_height, im_width), dtype=np.uint8)
            res = cv2.bitwise_and(mask, bbox_mask, mask=ones_mask)

            # Compute the overlapping area
            overlap = np.count_nonzero(res)

            if overlap > best_overlap:
                best_bbox = face_info
                best_overlap = overlap

        # Result image after applying mask
        # mask_img = cv2.bitwise_and(image, image, mask=mask)
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
            #cv2.imshow('face extr query', face)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            ret.append(face)
            bbs_coord.append((x, y, _x, _y))
            # cv2.rectangle(mask_img, (x, y), (_x, _y), (0, 255, 0), 2)
            # result.append(mask_img)
        else:
            ret.append(None)
            bbs_coord.append(None)

    return ret, bbs_coord  # , result


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
    ret, bbs_coord = detect_face_by_image(query, masks)
    # for res, path in zip(result, query_path):
    # im_name = path.split('/')[-1]
    # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./test_output/detect_result/' + im_name, res)
    return zip(ret, query_path, masks_path), bbs_coord


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

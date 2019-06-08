from face_extraction_queries import detect_face_by_image
from face_extraction import extract_faces_from_image
from deep_learning_utils import extendBB

import cv2
import numpy as np
import json
import os
import sys
import tensorflow as tf


sys.path.append(os.path.abspath(os.path.join('..', '3rd_party')))
from ServiceMTCNN import detect_face as lib


def getCorrectFrameInShot(topic_frame_path, shot_path):
    topic_frame = cv2.imread(topic_frame_path)
    topic_frame = cv2.resize(topic_frame, (768, 576))

    cap = cv2.VideoCapture(shot_path)

    all_frames = []
    min_topic_offset = None
    min_diff = np.inf
    current_offset = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            all_frames.append(frame)
            diff = np.sum((topic_frame - frame) ** 2)
            if diff < min_diff:
                min_topic_offset = current_offset
                min_diff = diff
        else:
            break

        current_offset += 1

    cap.release()
    return all_frames, min_topic_offset


def getCorrectFaceTrackInShot(topic_frame_path, topic_mask_path, shot_path):
    all_frames, topic_offset = getCorrectFrameInShot(
        topic_frame_path, shot_path)

    topic_frame = all_frames[topic_offset]
    mask_frame = cv2.imread(topic_mask_path, 0)
    mask_frame = cv2.resize(mask_frame, (768, 576))

    _, bb, _ = detect_face_by_image([topic_frame], [mask_frame])
    bb = list(bb[0])

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_topic_frame_gray = cv2.cvtColor(topic_frame, cv2.COLOR_BGR2GRAY)
    # old_topic_frame_gray = cv2.equalizeHist(old_topic_frame_gray)
    face_mask = np.zeros_like(old_topic_frame_gray)
    cv2.rectangle(face_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)
    
    face_mask = cv2.bitwise_and(face_mask, mask_frame)
    # face_mask = mask_frame
    # cv2.imshow('mask', face_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    p0 = cv2.goodFeaturesToTrack(
        old_topic_frame_gray, mask=face_mask, **feature_params)
    p0_temp = p0

    # Create session for MTCNN
    sess = tf.Session()
    pnet, rnet, onet = lib.create_mtcnn(sess, None)

    # Backward Tracking
    for idx, i in enumerate(range(topic_offset-1, -1, -1)):
        visualize_frame = all_frames[i].copy()
        frame_gray = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_topic_frame_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        avg_x = 0
        avg_y = 0
        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.circle(visualize_frame, (x_new, y_new), 2, (0, 255, 0), -1)

            avg_x += (x_new - x_old)
            avg_y += (y_new - y_old)
        avg_x = int(avg_x / len(good_new))
        avg_y = int(avg_y / len(good_new))

        # Update Bbox
        bb[0] += avg_x
        bb[1] += avg_y
        bb[2] += avg_x
        bb[3] += avg_y


        cv2.rectangle(visualize_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.imshow('forward tracking', visualize_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if idx != 0 and idx % 15 == 0:
        #     all_frames[i] = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB)
        #     print("here", all_frames[i].shape)
        #     _, bbs = extract_faces_from_image(all_frames[i], pnet, rnet, onet)
        #     all_frames[i] = cv2.cvtColor(all_frames[i], cv2.COLOR_RGB2BGR)
        #     
        #     temp = all_frames[i].copy()
        #     for bbox in bbs:
        #         cv2.rectangle(temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #     cv2.imshow('fig', temp)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()

        #     best_fit_bbox = None
        #     best_cnt_inlier = -np.inf 
        #     for bbox in bbs:
        #         cnt_inlier = 0
        #         for new in good_new:
        #             x, y = new.ravel()
        #             if x >= bbox[0] and x <= bbox[2] and y >= bbox[1] and y <= bbox[3]:
        #                 cnt_inlier += 1 

        #         if cnt_inlier != 0 and cnt_inlier > best_cnt_inlier:
        #             best_fit_bbox = bbox
        #             best_cnt_inlier = cnt_inlier

        #     bb = list(best_fit_bbox)
            face_mask = np.zeros_like(old_topic_frame_gray)
            cv2.rectangle(face_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)
            new_corner = cv2.goodFeaturesToTrack(
                frame_gray, mask=face_mask, **feature_params)

            good_new = good_new.tolist()
            new_corner = new_corner.squeeze().tolist()
            good_new.extend(new_corner)
            good_new = np.array(good_new, np.float32)
            

        old_topic_frame_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # foward tracking
    # p0 = p0_temp
    # for i in range(topic_offset+1, len(all_frames)):
    #     frame_gray = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)

    #     # calculate optical flow
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(
    #         old_topic_frame_gray, frame_gray, p0, None, **lk_params)

    #     good_new = p1[st == 1]
    #     good_old = p0[st == 1]


    #     avg_x = 0
    #     avg_y = 0
    #     for new, old in zip(good_new, good_old):
    #         x_new, y_new = new.ravel()
    #         x_old, y_old = old.ravel()
    #         cv2.circle(all_frames[i], (x_new, y_new), 2, (0, 255, 0), -1)

    #         avg_x += (x_new - x_old)
    #         avg_y += (y_new - y_old)
    #     avg_x = int(avg_x / len(good_new))
    #     avg_y = int(len(good_new))

    #     # Update Bbox
    #     print(bb)
    #     bb[0] += avg_x
    #     bb[1] += avg_y
    #     bb[2] += avg_x
    #     bb[3] += avg_y

    #     cv2.rectangle(all_frames[i], (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    #     cv2.imshow('forward tracking', all_frames[i])
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    #     old_topic_frame_gray = frame_gray.copy()
    #     p0 = good_new.reshape(-1, 1, 2)


def getCorrectFaceTrackInShotDense(topic_frame_path, topic_mask_path, shot_path):
    all_frames, topic_offset = getCorrectFrameInShot(
        topic_frame_path, shot_path)

    topic_frame = all_frames[topic_offset]
    mask_frame = cv2.imread(topic_mask_path, 0)
    mask_frame = cv2.resize(mask_frame, (768, 576))

    _, bb, _ = detect_face_by_image([topic_frame], [mask_frame])
    bb = list(bb[0])
    print('Bbox', bb)


    old_topic_frame_gray = cv2.cvtColor(topic_frame, cv2.COLOR_BGR2GRAY)
    # old_topic_frame_gray = cv2.equalizeHist(old_topic_frame_gray)

    face_mask = np.zeros_like(old_topic_frame_gray)
    cv2.rectangle(face_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)
    face_mask = cv2.bitwise_and(face_mask, mask_frame)
    # face_mask = mask_frame
    old_face_px_coord = np.where(face_mask == 255)
    old_face_px_coord = list(zip(old_face_px_coord[1], old_face_px_coord[0]))
    print(old_face_px_coord)


    # Create session for MTCNN
    # sess = tf.Session()
    # pnet, rnet, onet = lib.create_mtcnn(sess, None)

    # Backward Tracking
    for idx, i in enumerate(range(topic_offset-1, -1, -1)):
        visualize_frame = all_frames[i].copy()
        frame_gray = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)

        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(old_topic_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


        new_face_px_coord = []
        height, width = frame_gray.shape[:2]
        for px in old_face_px_coord:
            oldx, oldy = px
            if oldx >= width:
                oldx = width - 1
            if oldy >= height:
                oldy = height -1 

            newx = int(oldx + flow[oldy, oldx][0])
            newy = int(oldy + flow[oldy, oldx][1])
            if newx >= width:
                newx = width - 1
            if newy >= height:
                newy = height - 1
            new_face_px_coord.append((newx, newy))
        
        for px in new_face_px_coord:
            cv2.circle(visualize_frame, px, 2, (0, 255, 0), -1)
        
        cv2.imshow('fig', visualize_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        # Update
        old_face_px_coord = new_face_px_coord 
        old_topic_frame_gray = frame_gray.copy()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    query_folder = cfg['raw_data']['queries_folder']
    query_shot_folder = cfg['raw_data']['shot_example_folder']

    test_topic_frame_path = os.path.join(query_folder, 'chelsea.3.src.png')
    test_topic_mask_path = os.path.join(query_folder, 'chelsea.3.mask.png')
    test_shot_path = os.path.join(
        query_shot_folder, 'chelsea', 'shot0_1146.mp4')
    topic_frame = cv2.imread(test_topic_frame_path)
    topic_frame = cv2.resize(topic_frame, (768, 576))

    all_frames, topic_offset = getCorrectFrameInShot(
        test_topic_frame_path, test_shot_path)

    # cv2.imshow('found frame', all_frames[topic_offset])
    # cv2.imshow('gt frame', topic_frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    getCorrectFaceTrackInShot(test_topic_frame_path,
                              test_topic_mask_path, test_shot_path)

import sys
import os
import numpy as np
import cv2
import json
import tensorflow as tf
import glob
sys.path.append('../3rd_party')
from ServiceMTCNN import detect_face as lib

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
sess = tf.Session()
pnet, rnet, onet = lib.create_mtcnn(sess, None)
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
print("[+] Initialized MTCNN modules")

frames = glob.iglob('test_faces/**/**/*.jpg')
for frame in frames:
    f_id = frame.split('/')[-1].split('.')[0]
    img = cv2.imread(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, landmarks = lib.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    for index, box in enumerate(boxes):
            if box[4] < 0.9:  # remove faces with low confidence
                continue
            x1, y1, x2, y2 = int(box[0]), int(
                box[1]), int(box[2]), int(box[3])
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            if y2 > img.shape[0]:
                y2 = img.shape[0]
            if x2 > img.shape[1]:
                x2 = img.shape[1]
            print("\t\tScore : ", box[4])
            a = np.copy(img)
            cv2.rectangle(a, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.imwrite('test_faces_res/' + f_id + '_' + str(box[4]) + '.jpg', a)

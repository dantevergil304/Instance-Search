import json
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', '3rd_party')))
from ServiceMTCNN import detect_face as lib
import pickle
import numpy as np
import tensorflow as tf

print("[+] Load config file")
with open("../cfg/config.json","r") as f:
    config = json.load(f)

frames_folder = os.path.abspath(config["processed_data"]["frames_folder"])
faces_folder = os.path.abspath(config["processed_data"]["faces_folder"])

sess = tf.Session()
pnet, rnet, onet = lib.create_mtcnn(sess, None)
minsize=20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
print("[+] Initialized MTCNN modules")


print("[+] Accessed frames folder : %s" % (frames_folder))

for shot in os.listdir(frames_folder):    
    shot_path = os.path.join(frames_folder, shot)
    print("\t[+] Accessed shot %s folder : %s" % (shot, shot_path))
    save_path = os.path.join(faces_folder, shot + ".pickle")  
    faces = list()
    for frame in os.listdir(shot_path):            
        img = cv2.imread(os.path.join(shot_path, frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        print("\t\t[+] Loaded frames %s " % (frame))
        boxes, _ = lib.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(boxes) > 0:
            print("\t\t[+] Detected faces in frame")        
        for i,box in enumerate(boxes):
            x1, y1, x2, y2  = int(box[0]), int(box[1]), int(box[2]), int(box[3]) 
            if y1 < 0: y1 = 0
            if x1 < 0: x1 = 0
            if y2 > img.shape[0]: h = img.shape[0] - y1
            if x2 > img.shape[1]: w = img.shape[1] - x1
            face = img[y1 : y2 , x1 : x2]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
            #cv2.imshow("face " + str(i), face)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()   
            faces.append(face)     
    if len(faces) > 0:               
        with open(save_path, 'wb')  as f:
            pickle.dump(faces, f)        
    print("\t\t[+] Saved faces to %s" % (save_path))



            




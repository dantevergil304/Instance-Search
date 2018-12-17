import json
import cv2
from mtcnn.mtcnn import MTCNN
import os
import pickle

print("[+] Load config file")
with open("../cfg/config.json","r") as f:
    config = json.load(f)

frames_folder = os.path.abspath(config["processed_data"]["frames_folder"])
faces_folder = os.path.abspath(config["processed_data"]["faces_folder"])
detector = MTCNN()

print("[+] Access frames folder : %s" % (frames_folder))


for shot in os.listdir(frames_folder):    
    shot_path = os.path.join(frames_folder, shot)
    print("\t[+] Access shot %s folder : %s" % (shot, shot_path))
    save_path = os.path.join(faces_folder, shot + ".pickle")  
    faces = list()
    for frame in os.listdir(shot_path):            
        img = cv2.imread(os.path.join(shot_path, frame))        
        print("\t\t[+] Loaded frames %s " % (frame))
        detected_faces = detector.detect_faces(img)            
        if len(detected_faces) > 0:
            print("\t\t[+] Detected faces in frame")        
        for face in detected_faces:
            x, y, w, h = face['box']
            face = img[y : y + h , x : x + w]
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
            #cv2.imshow("res", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()   
            faces.append(face)     
    if len(faces) > 0:               
        with open(save_path, 'wb')  as f:
            pickle.dump(faces, f)
    print("\t\t[+] Saved faces")

            




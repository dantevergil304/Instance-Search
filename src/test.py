from search import SearchEngine
from face_extraction_queries import detect_face_by_path
from default_feature_extraction import extract_feature_from_face
from apply_super_resolution import apply_super_res

import os
import cv2
import numpy as np
import json
import pickle 

def test_remove_bad_face(name):    
    query_folder = "../data/raw_data/queries/"
    query_img = [
        name + ".1.src.bmp",
        name + ".2.src.bmp",
        name + ".3.src.bmp",
        name + ".4.src.bmp query_folder = "../data/raw_data/queries/"
    query_img = [
        name + ".1.src.bmp",
        name + ".2.src.bmp",
        name + ".3.src.bmp",
        name + ".4.src.bmp"
        
    ]
    query_mask = [
        name + ".1.mask.bmp",
        name + ".2.mask.bmp",
        name + ".3.mask.bmp",
        name + ".4.mask.bmp"
    ]"
        
    ]
    query_mask = [
        name + ".1.mask.bmp",
        name + ".2.mask.bmp",
        name + ".3.mask.bmp",
        name + ".4.mask.bmp"
    ]

    search_eng = SearchEngine()
    query_img = [os.path.join(query_folder, img) for img in query_img]
    query_mask = [os.path.join(query_folder, mask) for mask in query_mask]
    # Detect
    faces = detect_face_by_path(query_img, query_mask)

    temp = [cv2.resize(face, (448, 448)) for face in faces]
    before = np.hstack(tuple(temp))
    cv2.imshow("before apply super-resolution", before)

    # Apply Super-res
    hires_faces = [] 
    for face in faces:
        res = apply_super_res(face) 
        hires_faces.append(res)
    temp_2 = [cv2.resize(hires_face, (448, 448)) for hires_face in hires_faces]
    after = np.hstack(tuple(temp_2))
    cv2.imshow("after apply super-resolution", after)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Extract VGG16 features
    img_features = []
    for face in hires_faces:
        feature = extract_feature_from_face(search_eng.default_vgg, face)
        img_features.append((face, feature))

    temp = [cv2.resize(face, (224, 224)) for face in faces]
    before = np.hstack(tuple(temp))
    cv2.imshow("before remove bad faces", before)
        
    query_new = search_eng.remove_bad_faces(img_features)
    temp_2 = [cv2.resize(query[0], (224, 224)) for query in query_new]        
    print("Total \"good\" faces : ", len(query_new))
    after = np.hstack(tuple(temp_2))
    cv2.imshow("after remove bad faces", after)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./test_output/" + name + "_before.jpg", before)
    cv2.imwrite("./test_output/" + name + "_after.jpg", after)


def get_statistic_data():
    with open("../cfg/config.json") as f:
        cfg = json.load(f)
    faces_folder = cfg["processed_data"]["faces_folder"]
    total = 0
    n = len(os.listdir(faces_folder))    
    for file in os.listdir(faces_folder):
        if os.path.isdir(os.path.join(faces_folder, file)):
            continue
        with open(os.path.join(faces_folder, file), "rb") as f:
            faces = pickle.load(f)
            total += len(faces)
    print("Total faces : ", total)
    print("Average : ", total / n)


if __name__ == '__main__':
    names=["archie", "billy", "ian", "janine", "peggy", "phil", "ryan", "shirley"]
    for name in names:
        test_remove_bad_face(name)
    # get_statistic_data()



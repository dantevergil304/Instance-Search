from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras.engine import Model
import json
import os
import pickle
import numpy as np
import cv2



def extract_feature_from_face(model, face):
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float64)
    face = np.expand_dims(face, axis=0)
    face = utils.preprocess_input(face, version=1)
    features = model.predict(face)
    print("\t\tShape : ", features.shape)
    return features    

def extract_database_faces_features(model, faces_folder, default_feature_folder):
    for faces_file in os.listdir(faces_folder):
        if os.path.isdir(os.path.join(faces_folder, faces_file)):
            continue
        save_path = os.path.join(default_feature_folder, faces_file)
        print("[+] Accessed faces from shot %s" % (faces_file.split(".")[0]))
        with open(os.path.join(faces_folder, faces_file), "rb") as f:
            faces = pickle.load(f)
        features_v = []
        for i, face in enumerate(faces):
            print("\t[+] Accessed face %d "% (i))            
            features_v.append(extract_feature_from_face(model, face))
        with open(save_path, "wb") as f:
            pickle.dump(features_v, f)
        print("\t[+] Saved extracted features to : %s \n"% (save_path))
        
if __name__ == '__main__':
    with open("../cfg/config.json", "r") as f:
        cfg = json.load(f)
    print("[+] Loaded config file")

    default_feature_folder = os.path.abspath(cfg["features"]["VGG_default_features"])
    faces_folder = os.path.abspath(cfg["processed_data"]["faces_folder"])

    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    out = vgg_model.get_layer("fc6").output
    model = Model(vgg_model.input, out)
    print("[+] Loaded VGGFace model")
    extract_database_faces_features(model, faces_folder, default_feature_folder)

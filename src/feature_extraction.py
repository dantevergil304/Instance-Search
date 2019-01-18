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
    '''
    Parameters:
    - model: a VGG-Face model for extracting features from face
    - face: a face image
    Returns:
    - features: feature vector of an input face
    '''
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float64)
    face = np.expand_dims(face, axis=0)
    face = utils.preprocess_input(face, version=1)
    features = model.predict(face)
    return features


def extract_database_faces_features(model, frames_folder, faces_folder, feature_folder):
    '''
    Extract feature for faces. Each shot will be save on disk and
    represented by a list whose element is a feature vector. Each element
    in this list corresponds to the element at the same position in the face
    list extracted earlier.

    Parameters:
    - model: a VGG-Face model for extracting features from face
    - frames_folder: path to folder of shot folders, each shot folder
    contains many frames
    - faces_folder: path to folder of extracted face. Detected faces in a shot
    are represented by a list whose element is a tuple (frame_name, (x1, y1, x2, y2))
    - feature_folder: path to folder used for saving face features
    '''
    i = 0
    faces_files = [(file, os.path.join(faces_folder, file))
                   for file in os.listdir(faces_folder)]

    for faces_file in faces_files:
        file_name = faces_file[0]
        file_path = faces_file[1]
        shot = file_name.split('.')[0]
        if os.path.isdir(file_path):
            continue

        save_path = os.path.join(feature_folder, file_name)
        if os.path.exists(save_path):
            print("\t\t" + file_name + ' has already existed')
            continue

        print("[+] id % d Accessed faces from %s" % (i + 1, shot))
        i = i + 1

        with open(file_path, "rb") as f:
            faces_data = pickle.load(f)

        features_v = []
        for index, face_data in enumerate(faces_data):
            frame_id = face_data[0]
            print("\t\t[+] Frame : %s" % (frame_id))
            x1, y1, x2, y2 = face_data[1]
            frame_img = cv2.imread(os.path.join(frames_folder, shot, frame_id))
            face_img = frame_img[y1: y2, x1: x2]
            #cv2.imshow(frame_id, face_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            features_v.append(extract_feature_from_face(model, face_img))
        print("\t\tTotal feature : %d" % (len(features_v)))
        with open(save_path, "wb") as f:
            pickle.dump(features_v, f)
        print("\t\t[+] Saved extracted features to : %s \n" % (save_path))


if __name__ == '__main__':
    with open("../cfg/config.json", "r") as f:
        cfg = json.load(f)
    print("[+] Loaded config file")

    default_feature_folder = os.path.abspath(
        cfg["features"]["VGG_default_features"])
    faces_folder = os.path.abspath(cfg["processed_data"]["faces_folder"])
    frames_folder = os.path.abspath(cfg["processed_data"]["frames_folder"])
    vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
    out = vgg_model.get_layer("fc6").output
    model = Model(vgg_model.input, out)
    print("[+] Loaded VGGFace model")
    extract_database_faces_features(
        model, frames_folder, faces_folder, default_feature_folder)

from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras.engine import Model
import numpy as np
from default_feature_extraction import extract_feature_from_face
import json
import pickle
import os 

def cosine_similarity(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001    
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))


class SearchEngine(object):
    def __init__(self):
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer("fc6").output        
        self.default_vgg = Model(vgg_model.input, out)        
        with open("../cfg/config.json", "r") as f:
                self.cfg = json.load(f)    
        with open("../cfg/search_config.json", "r") as f:
                self.search_cfg = json.load(f)
        self.default_feature_folder = os.path.abspath(self.cfg["features"]["VGG_default_features"])
        self.faces_folder = os.path.abspath(self.cfg["processed_data"]["faces_folder"])

    def remove_bad_faces(self, query_features):        
        n = len(query_features)
        confs = [0] * n
        query_final = []

        for i in range(n):            
            confs[i] = sum([cosine_similarity(query_features[i][1], query_features[j][1]) for j in range(n) if i != j])

        for i in range(n):                        
            mean = sum([confs[j] for j in range(n) if i != j]) / (n - 1)
            if confs[i] + 0.05 >= mean:
                query_final.append(query_features[i])
        if len(query_final) == 0:
            print("[!] ERROR : length of query is zero")        
            return query_features     #In case all faces are "bad" faces, return the same query features        
        return query_final  #Return list of features of "good" faces

    def mean_max_similarity(self, query, shot_faces): #mean for query , max for shot
        final_sim = 0;
        for q_face in query:
                max_sim = max([cosine_similarity(q_face, shot_face) for shot_face in shot_faces])
                final_sim += max_sim
        return final_sim / len(query)
                                            
    def stage_1(self, query):
        result = []
        for shot in os.listdir(self.default_feature_folder):
            shot_path = os.path.join(self.default_feature_folder, shot)
            with open(shot_path, "rb") as f:
                shot_faces_feat = pickle.load(f)
            sim = mean_max_similarity(query, shot_faces_feat)
            result.append((shot, sim, shot_faces_feat))
        result.sort(reverse=True, key= lambda x : x[1])
        result = result[:1000]            
        training_data = list()
        for i,data in enumerate(result):
            with open(os.path.join(self.faces_folder, data[0]), "rb") as f:
                X = pickle.load(f)
            label = i < 100
            Y = [label] * len(faces)
            training_data.extend(list(zip(X, Y)))
        return training_data

    def stage_2(self, query):
        pass

    def stage_3(self, query):
        pass

    def searching(self, query, mask):
        img_features = []        
        for img in query:
            face = detect_face_in_query()
            feature = extract_feature_from_face(face)
            img_features.append((faces, feature))
        query = self.remove_bad_faces(img_features)
        #result_stage_1 = self.stage_1(query)            

if __name__ == '__main__':
   pass
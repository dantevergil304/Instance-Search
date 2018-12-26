from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras.engine import Model
from default_feature_extraction import extract_feature_from_face
from apply_super_resolution import apply_super_res
from face_extraction_queries import detect_face_by_path
import numpy as np
import json
import pickle
import os 
import cv2
import time

def cosine_similarity(vector_a, vector_b):
    l2_vector_a = np.linalg.norm(vector_a) + 0.001
    l2_vector_b = np.linalg.norm(vector_b) + 0.001    
    return np.dot((vector_a / l2_vector_a), (vector_b.T / l2_vector_b))


class VisualizeTools():
    def __init__(self):
        pass
    def visualize_images_matrix(self, matrix_images, title, save_path=None):
        imgs = []
        for row in matrix_images:
            h = np.hstack(tuple([cv2.resize(img, (341, 192)) for img in row]))            
            imgs.append(h)
        img = np.vstack(tuple(imgs))
        if save_path:
            cv2.imwrite(save_path, img)
        #cv2.imshow(title, img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

class SearchEngine(object):
    def __init__(self, visualize_tool):
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer("fc6").output        
        self.default_vgg = Model(vgg_model.input, out)                
        with open("../cfg/config.json", "r") as f:
                self.cfg = json.load(f)    
        with open("../cfg/search_config.json", "r") as f:
                self.search_cfg = json.load(f)

        self.default_feature_folder = os.path.abspath(self.cfg["features"]["VGG_default_features"])
        self.faces_folder = os.path.abspath(self.cfg["processed_data"]["faces_folder"])
        self.result_images_folder = os.path.abspath(self.cfg["result"]["result_images"])
        self.visualize_tool = visualize_tool

    def view_result(self, query, result):
        for i,shot in enumerate(result):           
            shot_name = shot[0].split(".")[0]
            print("\tTop %d , Shot %s, Similarity %f" % (i + 1, shot_name, shot[1]))            
            #print("\t\tViewing top 3 faces in shot...")            
            matrix_faces = shot[2]
            n = 3 if len(matrix_faces[0]) > 3 else len(matrix_faces[0])            
            imgs = []
            imgs_v = [cv2.imread(q[0][1]) for q in query]
            masks = [cv2.imread(q[0][2]) for q in query]
            q_v = list(zip(imgs_v, masks))            
            for img_mask in q_v:
                imgs.append([img_mask[0], img_mask[1]])                
                        
            for index,faces in enumerate(matrix_faces):
                imgs[index].extend([face[0] for face in faces][:n])

            file_name = "top" + str(i+1) + "_" + shot_name + "_" + str(shot[1][0][0]) + ".jpg"
            save_path = os.path.join(self.result_images_folder, file_name)
            self.visualize_tool.visualize_images_matrix(imgs, shot_name, save_path=save_path)

    def remove_bad_faces(self, query):        
        n = len(query)
        confs = [0] * n
        query_final = []

        for i in range(n):            
            confs[i] = sum([cosine_similarity(query[i][1], query[j][1]) for j in range(n) if i != j])

        for i in range(n):                        
            mean = sum([confs[j] for j in range(n) if i != j]) / (n - 1)
            if confs[i] + 0.05 >= mean:
                query_final.append(query[i])
        if len(query_final) == 0:
            print("[!] ERROR : length of query is zero")        
            return query     # In case all faces are "bad" faces, return the same query features        
        return query_final  # Return list of features of "good" faces

    def form_training_set(self, result, thresh = 0.5):
        X = []
        Y = []
        for record in result:
            faces_matrix = record[2]
            for faces_row in faces_matrix:
                for face in faces_row:
                    X.append(face[0])
                    if face[1] > thresh: # Compare similarity between query of a query face and face to threshold
                        Y.append(1)
                    else:
                        Y.append(0)
        return (X, Y)



    def fine_tuning(self, training_set):
        X, Y = training_set[0], training_set[1]
        self.fine_tuned_model = 
    
    def mean_max_similarity(self, query, shot_faces): # mean for query , max for shot
        final_sim = 0;
        shot_faces_with_sim = []        
        for q_face in query:
                faces_sim = [(shot_face[0], cosine_similarity(q_face[1], shot_face[1]))  for shot_face in shot_faces]
                faces_sim.sort(reverse=True, key=lambda x : x[1])
                final_sim += faces_sim[0][1]
                shot_faces_with_sim.append(faces_sim) # Each image q_face in query have a list of corresponding faces which sorted based on similarity between faces and q_face. Overall, it a matrix of faces (1)
        return final_sim / len(query), shot_faces_with_sim                                                

    def stage_1(self, query):
        print("[+] Stage 1 of searching  : ")
        result = []
        shots = os.listdir(self.default_feature_folder)
        for shot in shots:            
            shot_path = os.path.join(self.default_feature_folder, shot)
            face_path = os.path.join(self.faces_folder, shot)
            with open(shot_path, "rb") as f:
                shot_faces_feat = pickle.load(f)
            with open(face_path, "rb") as f:
                shot_faces = pickle.load(f)
            shot_faces = list(zip(shot_faces, shot_faces_feat)) # shot faces is a list with elements consist of  (face matrix, face features)
            print("\tShot %s , number of faces : %d" % (shot.split(".")[0], len(shot_faces)))            
            sim, shot_faces_with_sim = self.mean_max_similarity(query, shot_faces)
            result.append((shot, sim, shot_faces_with_sim)) # Result is a list of  elements consist of (shot_id, similarity(query, shot_id), corresponding matrix faces like explaination (1)

        result.sort(reverse=True, key= lambda x : x[1])
        top = self.search_cfg["top_samples_for_training"]
        result = result[:top] 
        print("[+] Search completed, visualizing results")
        self.view_result(query, result)
        return result


    def stage_2(self, query, training_set):
        self.fine_tuning


    def stage_3(self, query):
        pass


    def searching(self, query, mask):      
        start = time.time()   
        query_faces = detect_face_by_path(query, mask)        
        print("[+] Detected faces from query")

        #imgs_v = [cv2.imread(q) for q in query]       
        #self.visualize_tool.visualize_images([imgs_v, query_faces], "Detected faces in query")       
        #####

        query_faces_sr = [(apply_super_res(face[0]), face[1], face[2]) for face in query_faces]  # storing a list of elements. Each elements including (query_faces matrix after sr, path to face's root image, path to face's root mask)
        #self.visualize_tool.visualize_images([query_faces, query_faces_sr],"Apply super resolution")
        print("[+] Applied super resolution to query")
        #####

        faces_features = []
        for face in query_faces_sr:            
            feature = extract_feature_from_face(self.default_vgg, face[0])
            faces_features.append((face, feature)) # faces_features store extractly like query_faces_sr except with addtional information, featuer of query faces
        print("[+] Extracted feature of query images")
        #####

        query_faces = self.remove_bad_faces(faces_features)
        # This is for visualization the query after remove bad faces        
        shift = len(query_faces_sr) - len(query_faces)
        temp = [query_face[0][0] for query_face in query_faces]        
        for i in range(shift):
            temp.append(np.zeros((224, 224,3), dtype=np.uint8))     
        #self.visualize_tool.visualize_images([query_faces_sr, temp], "Remove bad faces")
        print("[+] Removed bad faces from query")
        #####

        result = self.stage_1(query_faces)        
        end = time.time()
        print("[+] Execution time ", end-start , " second")                  
        training_set = self.form_training_set(result)
        result = self.stage_2(query_faces, training_set)
    

if __name__ == '__main__':
    query_folder = "../data/raw_data/queries/"
    names=["archie", "billy", "ian", "janine", "peggy", "phil", "ryan", "shirley"]
    name = names[1]
    query = [
        name + ".1.src.bmp",
        name + ".2.src.bmp",
        name + ".3.src.bmp",
        name + ".4.src.bmp"
        
    ]
    masks = [
        name + ".1.mask.bmp",
        name + ".2.mask.bmp",
        name + ".3.mask.bmp",
        name + ".4.mask.bmp"
    ]    
    
    query = [os.path.join(query_folder, q) for q in query]
    masks = [os.path.join(query_folder, m) for m in masks]
    print("[+] Query : ", names)
    imgs_v = [cv2.imread(q) for q in query]
    masks_v = [cv2.imread(m) for m  in masks]    
    visualize_tool = VisualizeTools()
    #visualize_tool.visualize_images([imgs_v, masks_v], "Query : "+ name)
    search_eng = SearchEngine(visualize_tool)
    print("[+] Initialized searh engine")
    search_eng.searching(query, masks)


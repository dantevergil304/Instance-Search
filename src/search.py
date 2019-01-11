from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras.engine import Model
from feature_extraction import extract_feature_from_face, extract_database_faces_features
from apply_super_resolution import apply_super_res
from face_extraction_queries import detect_face_by_path
from vgg_finetune import fine_tune
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

    def visualize_images(self, matrix_images, title, save_path=None, size=(341, 192)):
        imgs = []
        for row in matrix_images:
            h = np.hstack(tuple([cv2.resize(img, size) for img in row]))
            imgs.append(h)
        img = np.vstack(tuple(imgs))
        if save_path:
            cv2.imwrite(save_path, img)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class SearchEngine(object):
    def __init__(self, visualize_tool):
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer("fc6").output
        self.default_vgg = Model(vgg_model.input, out)
        with open("../cfg/config.json", "r") as f:
            self.cfg = json.load(f)
        with open("../cfg/search_config.json", "r") as f:
            self.search_cfg = json.load(f)

        self.default_feature_folder = os.path.abspath(
            self.cfg["features"]["VGG_default_features"])
        self.faces_folder = os.path.abspath(
            self.cfg["processed_data"]["faces_folder"])
        self.result_images_folder = os.path.abspath(
            self.cfg["result"]["result_images"])
        self.frames_folder = os.path.abspath(
            self.cfg["processed_data"]["frames_folder"])
        self.visualize_tool = visualize_tool
        self.fine_tune_vgg = None

    def view_result(self, query, result):
        print("[+] Visualize results")
        for i, record in enumerate(result):
            shot_id = record[0]
            print("\tTop %d , Shot %s, Similarity %f" %
                  (i + 1, shot_id, record[1]))
            #print("\t\tViewing top 3 faces in shot...")
            frames = self.calculate_average_faces_sim(record)
            n = 5 if len(frames) > 5 else len(frames)
            #imgs = []
            #imgs_v = [cv2.imread(q[0][1]) for q in query]
            #masks = [cv2.imread(q[0][2]) for q in query]
            #q_v = list(zip(imgs_v, masks))
            # for img_mask in q_v:
            #    imgs.append([img_mask[0], img_mask[1]])

            # for index,row in enumerate(matrix_frames):
            #    frames_with_faces = []
            #    frames = sorted(row, reverse=True, key=lambda x: x[1])
            #    frames = frames[:n]
            #    for frame in frames:
            #        name = frame[0][0]
            #        img = cv2.imread(os.path.join(self.frames_folder, shot_id, name))
            #        x1, y1, x2, y2 = frame[0][1]
            #        cv2.rectangle(img, (x1, y1), (x2,y2), (0, 255, 0), 2)
            #        frames_with_faces.append(img)
            #    imgs[index].extend(frames_with_faces)
            frames = sorted(frames, reverse=True, key=lambda x: x[1])
            frames = frames[:n]
            frames_with_faces = []
            for index, frame in enumerate(frames):
                name = frame[0][0]
                img = cv2.imread(os.path.join(
                    self.frames_folder, shot_id, name))
                x1, y1, x2, y2 = frame[0][1]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.imshow("none", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                frames_with_faces.append(img)

            file_name = "top" + str(i+1) + "_" + \
                shot_id + "_" + str(record[1][0]) + ".jpg"
            save_path = os.path.join(self.result_images_folder, file_name)
            self.visualize_tool.visualize_images(
                [frames_with_faces], shot_id, save_path=save_path)

    def remove_bad_faces(self, query):
        n = len(query)
        confs = [0] * n
        query_final = []

        for i in range(n):
            confs[i] = sum([cosine_similarity(query[i][1], query[j][1])
                            for j in range(n) if i != j])

        for i in range(n):
            mean = sum([confs[j] for j in range(n) if i != j]) / (n - 1)
            if confs[i] + 0.05 >= mean:
                query_final.append(query[i])
        if len(query_final) == 0:
            print("[!] ERROR : length of query is zero")
            return query     # In case all faces are "bad" faces, return the same query features
        return query_final  # Return list of features of "good" faces

    def calculate_average_faces_sim(self, record):
        '''PARAMETER:
        - record: a list contains 3 elements,
        1st elements: shotID,
        2nd elements: similarity score,
        3rd elements: result matrix'''
        faces_data = []
        faces_matrix = record[2]
        col = len(faces_matrix[0])
        row = len(faces_matrix)
        for i in range(col):
            data = faces_matrix[0][i][0]
            mean_sim = 0
            for j in range(row):
                mean_sim += faces_matrix[j][i][1]
            mean_sim /= row
            faces_data.append((data, mean_sim))
        return faces_data

    def split_by_average_comparision(self, record, thresh=0.5):
        X = []
        Y = []
        pos, neg = 0, 0
        shot_id = record[0]
        mean_sim = record[1]
        print("\t\tNumber of faces in %s : %d" % (shot_id, len(record[2][0])))
        data = self.calculate_average_faces_sim(record)
        for face_data in data:
            img = cv2.imread(os.path.join(
                self.frames_folder, shot_id, face_data[0][0]))
            x1, y1, x2, y2 = face_data[0][1]
            face = img[y1: y2, x1: x2]
            X.append(face)
            if face_data[1] >= thresh:
                Y.append(1)
                pos += 1
            else:
                Y.append(0)
                neg += 1
        return X, Y, pos, neg
    '''
    def split_by_thresh_comparision(self, record, thresh=0.6):
        X = []
        Y = []
        pos, neg = 0, 0
        shot_id = record[0]
        mean_sim = record[1]
        print("\t\tNumber of faces in %s : %d" % (shot_id, len(record[2][0])))
        data = record[2]
        pos_sample = set()                
        neg_sample = set()
        for row in data:
            for face_data in row:
                if face_data[1] >= thresh:
                    x_sample.add(face_data)                
        for face_data in x_sample:
            img = cv2.imread(os.path.join(self.frames_folder, shot_id, face_data[0][0]))
            x1, y1, x2, y2 = face_data[0][1]
            face = img[y1 : y2, x1 : x2]
            X.append(face)
            if face_data[1] >= thresh:                    
                Y.append(1)
                pos +=1 
            else:
                Y.append(0)
                neg += 1                
            return X, Y, pos, neg
    '''

    def form_training_set(self, result):
        X = []
        Y = []
        pos = 0
        neg = 0
        print("\t[+] Forming training set...")
        for record in result:
            x_sample, y_sample, pos_, neg_ = self.split_by_average_comparision(
                record)
            X.extend(x_sample)
            Y.extend(y_sample)
            pos += pos_
            neg += neg_
        print("\t[+] Finished, There are %d positive sample and %d negative sample in top %d" %
              (pos, neg, len(result)))
        return (X, Y)

    def mean_max_similarity(self, query, shot_faces):  # mean for query , max for shot
        final_sim = 0
        frames_with_bb_sim = []
        for q_face in query:
            faces_sim = [(shot_face[0], cosine_similarity(
                q_face[1], shot_face[1])) for shot_face in shot_faces]
            max_sim = max(faces_sim, key=lambda x: x[1])
            final_sim += max_sim[1]
            # Each image q_face in query have a list of corresponding faces which sorted based on similarity between faces and q_face. Overall, it a matrix of faces (1)
            frames_with_bb_sim.append(faces_sim)
        return final_sim / len(query), frames_with_bb_sim

    def stage_1(self, query):
        ''' Return
        List of elements, each consist of:
        - shot_id
        - sim(query, shot_id): similarity between input query and current shot 
        - a matrix of shape(num_query_face, num_shot_face_detected):
            + num_query_face: #remaining faces in query after remove bad faces
            + num_shot_face_detected: #faces detected of the current shot
            + matrix[i][j]: 
        '''
        print("[+] Stage 1 of searching: ")
        result = []
        shot_feature_files = [(file, os.path.join(self.default_feature_folder, file))
                              for file in os.listdir(self.default_feature_folder)]
        for shot_feature_file in shot_feature_files:
            shot_id = shot_feature_file[0].split(".")[0]
            feature_path = shot_feature_file[1]
            face_path = os.path.join(self.faces_folder, shot_feature_file[0])
            with open(feature_path, "rb") as f:
                shot_faces_feat = pickle.load(f)
            with open(face_path, "rb") as f:
                shot_faces = pickle.load(f)
            # shot faces is a list with elements consist of  ((frame, (x1, y1, x2, y2)), face features)
            shot_faces = list(zip(shot_faces, shot_faces_feat))
            print("\t%s , number of faces : %d" % (shot_id, len(shot_faces)))

            sim, frames_with_bb_sim = self.mean_max_similarity(
                query, shot_faces)
            # Result is a list of elements consist of (shot_id, similarity(query, shot_id), corresponding matrix faces like explaination (1)
            result.append((shot_id, sim, frames_with_bb_sim))

        result.sort(reverse=True, key=lambda x: x[1])
        top = self.search_cfg["top_samples_for_training"]
        result = result[:top]
        print("[+] Search completed")
        # self.view_result(query, result)
        return result

    def stage_2(self, query, training_set):
        #self.fine_tune_vgg = fine_tune(self.default_vgg, training_set)

        pass

    def stage_3(self, query):
        pass

    def searching(self, query, mask):
        start = time.time()
        query_faces = detect_face_by_path(query, mask)
        print("[+] Detected faces from query")

        #imgs_v = [cv2.imread(q) for q in query]
        #self.visualize_tool.visualize_images([imgs_v, query_faces], "Detected faces in query")
        #####

        # storing a list of elements. Each elements including (query_faces matrix after sr, path to face's root image, path to face's root mask)
        query_faces_sr = [(apply_super_res(face[0]), face[1], face[2])
                          for face in query_faces]
        #self.visualize_tool.visualize_images([query_faces, query_faces_sr],"Apply super resolution")
        print("[+] Applied super resolution to query")
        #####

        faces_features = []
        for face in query_faces_sr:
            feature = extract_feature_from_face(self.default_vgg, face[0])
            # faces_features store extractly like query_faces_sr except with addtional information, featuer of query faces
            faces_features.append((face, feature))
        print("[+] Extracted feature of query images")
        #####

        query_faces = self.remove_bad_faces(faces_features)
        # This is for visualization the query after remove bad faces
        shift = len(query_faces_sr) - len(query_faces)
        temp = [query_face[0][0] for query_face in query_faces]
        for i in range(shift):
            temp.append(np.zeros((341, 192, 3), dtype=np.uint8))
        # self.visualize_tool.visualize_images(
        #    [query_faces_sr, temp], "Remove bad faces")
        print("[+] Removed bad faces from query")
        #####

        result = self.stage_1(query_faces)
        end = time.time()
        print("[+] Execution time ", end-start, " second")
        training_set = self.form_training_set(result)

        #result = self.stage_2(query_faces, training_set)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    query_folder = "../data/raw_data/queries/"
    names = ["archie", "billy", "ian", "janine",
             "peggy", "phil", "ryan", "shirley"]
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
    masks_v = [cv2.imread(m) for m in masks]
    visualize_tool = VisualizeTools()
    visualize_tool.visualize_images([imgs_v, masks_v], "Query : " + name)
    search_eng = SearchEngine(visualize_tool)
    print("[+] Initialized searh engine")
    search_eng.searching(query, masks)

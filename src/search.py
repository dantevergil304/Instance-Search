from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras.engine import Model
from keras import backend as K
from feature_extraction import extract_feature_from_face, extract_database_faces_features
from apply_super_resolution import apply_super_res
from face_extraction_queries import detect_face_by_path
from vgg_finetune import fine_tune, extract_face_features
from sticher import ImageSticher
from keras.models import load_model
from util import calculate_average_faces_sim, cosine_similarity, mean_max_similarity, write_result_to_file, write_result
from scipy import stats
from PIL import Image
from poseEstimate import getFaceRotationAngles

import numpy as np
import json
import pickle
import os
import cv2
import time
import sys
import glob


class SearchEngine(object):
    def __init__(self, visualize_tool):
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer("fc6").output
        self.default_vgg = Model(vgg_model.input, out)
        with open("../cfg/config.json", "r") as f:
            self.cfg = json.load(f) 
        with open("../cfg/search_config.json", "r") as f:
            self.search_cfg = json.load(f)

        # Set up folder path
        self.default_feature_folder = os.path.abspath(
            self.cfg["features"]["VGG_default_features"])
        self.faces_folder = os.path.abspath(
            self.cfg["processed_data"]["faces_folder"])
        self.landmarks_folder = os.path.abspath(
            self.cfg["processed_data"]["landmarks_folder"])
        self.frames_folder = os.path.abspath(
            self.cfg["processed_data"]["frames_folder"])
        self.fine_tune_feature_folder = os.path.abspath(
            self.cfg["features"]["VGG_fine_tuned_features"])
        self.fine_tune_model_path = os.path.abspath(
            self.cfg["models"]["VGG_folder"]["VGG_fine_tuned_folder"])
        self.svm_model_path = os.path.abspath(
            self.cfg['models']["SVM_folder"])
        self.training_data_folder = os.path.abspath(
            self.cfg["training_data"]["data"])
        self.result_path = self.cfg["config_path"]

        self.query_name = None
        self.visualize_tool = visualize_tool
        self.fine_tune_vgg = None
        self.svm_clf = None

    def remove_bad_faces(self, query):
        '''
        Parameters:
        - query: [((face matrix, img query path, binary mask path), feature vector)]
        Returns:
        - query_final: resulting query after remove some bad faces. (same format at parameter)
        '''
        n = len(query)
        confs = [0] * n
        query_final = []

        for i in range(n):
            confs[i] = sum([cosine_similarity(query[i][1], query[j][1])
                            for j in range(n) if i != j])

        if n > 1:
            for i in range(n):
                mean = sum([confs[j] for j in range(n) if i != j]) / (n - 1)
                if confs[i] + 0.05 >= mean:
                    query_final.append(query[i])

        if len(query_final) == 0:
            print("[!] ERROR : No image in query")
            return query     # In case all faces are "bad" faces, return the same query features
        return query_final  # Return list of features of "good" faces

    def split_by_average_comparision(self, record, thresh=0.5):
        '''
        Specifiy if a sample is positive or negative based on
        given thresh. If mean similarity of that shot is larger than
        thresh, a sample is positive. Otherwise, it is negative.
        Parameters:
        record: a list contains 3 elements:
        - shot_id
        - sim(query, shot_id): similarity between input query and current shot 
        - a matrix of shape(num_query_face, num_shot_face_detected):
            + num_query_face: #remaining faces in query after remove bad faces
            + num_shot_face_detected: #faces detected of the current shot
            + matrix[i][j]: ((frame file, bb), cosine similarity score 
        between 'query face i' and 'shot face j')
        thresh: a similarity thresh for choosing pos and neg_

        Returns:
        X: a list of faces use for training
        y: a list of corresponding labels for each face
        pos: # positive samples
        neg: # neg samples
        '''
        X = []
        Y = []
        pos, neg = 0, 0
        shot_id = record[0]
        video_id = shot_id.split('_')[0][4:]
        data = calculate_average_faces_sim(record)
        for face_data in data:
            img = cv2.imread(os.path.join(
                self.frames_folder, 'video' + video_id, shot_id, face_data[0][0]))
            x1, y1, x2, y2 = face_data[0][1]
            face = img[y1: y2, x1: x2]
            face = cv2.resize(face, (224, 224))
            X.append(face)
            if face_data[1] >= thresh:
                Y.append(1)
                pos += 1
            else:
                Y.append(0)
                neg += 1
        return X, Y, pos, neg

    def _PEsolvePnP(self, x_sample, y_sample, landmarks_info):
        '''
        Eliminate side faces by using solvePnP for pose estimation
        '''
        new_x_sample = []
        new_y_sample = []
        pos = 0
        neg = 0
        rotation_vecs = []
        for image_points, (height, width) in landmarks_info:
            rotation_vecs.append(getFaceRotationAngles(
                image_points, (height, width)))
        for idx, (x_smp, y_smp) in enumerate(zip(x_sample, y_sample)):
            if abs(rotation_vecs[idx][0]) < 30 and abs(rotation_vecs[idx][1]) < 10:
                new_x_sample.append(x_smp)
                new_y_sample.append(y_smp)

        for x_smp, y_smp in zip(new_x_sample, new_y_sample):
            if y_smp == 1:
                pos += 1
            else:
                neg += 1

        # if len(x_sample) != len(new_x_sample):
        #     print('[+] # Samples before removing %d' % (len(x_sample)))
        #     print('[+] # Samples after removing %d' % (len(new_x_sample)))

        return new_x_sample, new_y_sample, pos, neg

    def form_training_set(self, result, thresh=0.7, rmBadFaces=None):
        X = []
        Y = []
        pos = 0
        neg = 0
        print("[+] Forming training set...")
        for record in result:
            shot_id = record[0]
            video_id = shot_id.split('_')[0][4:]
            with open(os.path.join(self.faces_folder, 'video' + video_id, shot_id + ".pickle"), 'rb') as f:
                faces = pickle.load(f)
            with open(os.path.join(self.landmarks_folder, 'video' + video_id, shot_id + ".pickle"), 'rb') as f:
                landmarks = pickle.load(f)

            # Get rotation vector of  each face in current shot
            landmarks_info = []
            for (frame_id, _), landmark in zip(faces, landmarks):
                img = Image.open(os.path.join(
                    self.frames_folder, 'video' + video_id, shot_id, frame_id))
                width, height = img.size

                image_points = []
                for i in range(int(len(landmark)/2.)):
                    x, y = int(landmark[i]), int(landmark[i+5])
                    image_points.append((x, y))
                image_points = np.array(image_points, dtype='double')

                # rotation_vecs.append(getFaceRotationAngles(
                #     image_points, (height, width)))
                landmarks_info.append((image_points, (height, width)))

            # Choose positive and negative sample
            x_sample, y_sample, pos_, neg_ = self.split_by_average_comparision(
                record, thresh=thresh)

            if rmBadFaces is not None:
                x_sample, y_sample, pos_, neg_ = rmBadFaces(
                    x_sample, y_sample, landmarks_info)

            X.extend(x_sample)
            Y.extend(y_sample)
            pos += pos_
            neg += neg_

        print("[+] Finished, There are %d positive sample and %d negative sample in top %d" %
              (pos, neg, len(result)))
        data = list(zip(X, Y))
        data.sort(reverse=True, key=lambda x: x[1])
        data = data[:pos + pos + pos]
        X = [a[0] for a in data]
        Y = [a[1] for a in data]

        filename = self.query_name + '_' + 'thresh=' + str(thresh)
        if rmBadFaces is not None:
            filename += rmBadFaces.__name__

        save_path = os.path.join(self.training_data_folder, self.query_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        training_set_path = os.path.join(
            save_path, filename + '_svm_dataset.pkl')
        with open(training_set_path, "wb") as f:
            pickle.dump([X, Y], f)
        self.visualize_tool.view_training_set(training_set_path)
        return [X, Y]

    def form_SVM_training_set(self, result, thresh=0.7, rmBadFaces=None):
        X, Y = self.form_training_set(result, thresh, rmBadFaces)
        filename = os.listdir(os.path.join(
            self.fine_tune_model_path, self.query_name))[0]
        model_path = os.path.join(
            self.fine_tune_model_path, self.query_name, filename)

        print('[+] Extracting face features')
        features = extract_face_features(model_path, X)
        print('[+] Finished extracting face features')

        K.clear_session()
        return [features, Y]

    def stage_1(self, query, feature_folder, isStage3=False):
        ''' 
        Parameters:
        - query: [((face matrix, img query path, binary mask path), feature vector)]
        - feature_folder: path to folder of features
        - top: the number of retrieval results

        Returns:
        List of elements, each consist of:
        - shot_id
        - sim(query, shot_id): similarity between input query and current shot 
        - a matrix of shape(num_query_face, num_shot_face_detected):
            + num_query_face: #remaining faces in query after remove bad faces
            + num_shot_face_detected: #faces detected of the current shot
            + matrix[i][j]: ((frame file, bb), cosine similarity score 
        between 'query face i' and 'shot face j')
        '''

        result = []
        print("[+] Current feature folder : ", feature_folder)
        # shot_feature_files = [(file, os.path.join(feature_folder, file))
        #                      for file in os.listdir(feature_folder)]
        shot_feature_files = [(os.path.basename(path), path) for path in glob.iglob(
            feature_folder + '/**/*.pickle', recursive=True)]        
        cosine_similarity = []
        classification_score = []        
        print('[+] Start to compute the similarity between person and each shot\n')
        for idx, shot_feature_file in enumerate(shot_feature_files):
            shot_id = shot_feature_file[0].split(".")[0]
            video_id = shot_id.split('_')[0][4:]
            print('[id: %d], computing similarity for shot id: %s' %
                  (idx, shot_id))
            feature_path = shot_feature_file[1]
            face_path = os.path.join(
                self.faces_folder, 'video' + video_id, shot_feature_file[0])
            with open(feature_path, "rb") as f:
                shot_faces_feat = pickle.load(f)
            with open(face_path, "rb") as f:
                shot_faces = pickle.load(f)
            # shot faces is a list with elements consist of  ((frame, (x1, y1, x2, y2)), face features)
            shot_faces = list(zip(shot_faces, shot_faces_feat))
            #print("\t%s , number of faces : %d" % (shot_id, len(shot_faces)))

            sim, frames_with_bb_sim = mean_max_similarity(
                query, shot_faces)

            if isStage3:
                arr = [self.svm_clf.decision_function(
                    face_feat) for face_feat in shot_faces_feat]
                decision_score = max(arr)
                exact_distance = decision_score / \
                    np.linalg.norm(self.svm_clf.coef_)

                cosine_similarity.append(sim)
                classification_score.append(np.expand_dims(exact_distance, 0))

            # Result is a list of elements consist of (shot_id, similarity(query, shot_id), corresponding matrix faces like explaination (1)
            result.append((shot_id, sim, frames_with_bb_sim))
        print('[+] Finished computing similarity for all shots')

        if isStage3:
            person_similarity = stats.zscore(
                cosine_similarity) + stats.zscore(classification_score)
            shot_id, _, frames_with_bb_sim = zip(*result)
            result = list(zip(shot_id, person_similarity, frames_with_bb_sim))

        result.sort(reverse=True, key=lambda x: x[1])                
        print("[+] Search completed")
        return result[:1000]

    def stage_2(self, query, training_set):
        '''
        Parameter:
        - query: [((face matrix, img query path, binary mask path), feature vector)]
        - training_set: a training set 
        '''
        print("[+] Begin stage 2 of searching")

        # self.fine_tune_vgg = fine_tune(
        #     training_set, model_name=self.query_name)
        # print("[+] Finished fine tuned VGG Face model")

        filename = os.listdir(os.path.join(
            self.fine_tune_model_path, self.query_name))[0]
        self.fine_tune_vgg = load_model(os.path.join(
            self.fine_tune_model_path, self.query_name, filename))
        print("[+] Loaded VGG fine tuned model")

        feature_extractor = Model(
            self.fine_tune_vgg.input, self.fine_tune_vgg.get_layer('fc6').output)

        fine_tune_feature_folder = os.path.join(
            self.fine_tune_feature_folder, self.query_name)
        if not os.path.isdir(fine_tune_feature_folder):
            os.makedirs(fine_tune_feature_folder)
            print("[+]Begin extract feature using fine tuned model")
            extract_database_faces_features(
                feature_extractor, self.frames_folder, self.faces_folder, fine_tune_feature_folder)
            print("[+] Finished extract feature")
        query_faces = []
        for face in query:
            # faces_features store extractly like query_faces_sr except with addtional information, feature of query faces
            feature = extract_feature_from_face(feature_extractor, face[0][0])
            query_faces.append((face[0], feature))

        K.clear_session()
        return self.stage_1(query_faces, fine_tune_feature_folder)

    def stage_3(self, query, training_set=None):
        from sklearn.svm import SVC

        X, y = training_set[0], training_set[1]

        print('[+] Begin Training SVM')
        self.svm_clf = SVC(probability=True, verbose=True,
                           random_state=42, kernel='linear')
        self.svm_clf.fit(X, y)
        print('[+] Fininshed Training SVM')

        with open(os.path.join(self.svm_model_path, self.query_name, self.query_name + '.pkl'), 'wb') as f:
            pickle.dump(self.svm_clf, f)

        # with open(os.path.join(self.svm_model_path, 'archie.pkl'), 'rb') as f:
        #     self.svm_clf = pickle.load(f)

        query_faces = []
        filename = os.listdir(os.path.join(
            self.fine_tune_model_path, self.query_name))[0]
        fine_tune_vgg = load_model(os.path.join(
            self.fine_tune_model_path, self.query_name, filename))
        feature_extractor = Model(
            fine_tune_vgg.input, fine_tune_vgg.get_layer('fc6').output)
        for face in query:
            # faces_features store extractly like query_faces_sr except with addtional information, feature of query faces
            feature = extract_feature_from_face(feature_extractor, face[0][0])
            query_faces.append((face[0], feature))

        K.clear_session()

        return self.stage_1(query_faces, os.path.join(self.fine_tune_feature_folder, self.query_name), isStage3=True)

    def searching(self, query, mask, isStage1=True, isStage2=False, isStage3=False):
        start = time.time()
        query_faces = detect_face_by_path(query, mask)
        print("[+] Detected faces from query")

        #imgs_v = [cv2.imread(q) for q in query]
        #self.visualize_tool.visualize_images([imgs_v, query_faces], "Detected faces in query")
        #####

        # storing a list of elements. Each elements including (query_faces matrix after sr, path to face's root image, path to face's root mask)
        query_faces_sr = [(apply_super_res(face[0]), face[1], face[2])
                          for face in query_faces]
        print("[+] Applied super resolution to query")
        #self.visualize_tool.visualize_images([query_faces, query_faces_sr],"Apply super resolution")
        #####

        faces_features = []
        for face in query_faces_sr:
            feature = extract_feature_from_face(self.default_vgg, face[0])
            # faces_features store extractly like query_faces_sr except with addtional information, featuer of query faces
            faces_features.append((face, feature))
        print("[+] Extracted feature of query images")
        #####

        query_faces = self.remove_bad_faces(faces_features)
        print("[+] Removed bad faces from query")
        # This is for visualization the query after remove bad faces
        """
        shift = len(query_faces_sr) - len(query_faces)
        temp = [query_face[0][0] for query_face in query_faces]
        for i in range(shift):
            temp.append(np.zeros((341, 192, 3), dtype=np.uint8))
        img_faces = [img[0] for img in query_faces_sr]
        self.visualize_tool.visualize_images(
            [img_faces, temp], "Remove bad faces", save_path="../result/" + self.query_name + "_rm_bad_faces.jpg")
        #####
        """
        if isStage1:
            print("\n==============================================================================")
            print("\n                       [+] Stage 1 of searching:\n")
            print(
                "==============================================================================")
            # Check if result at stage 1 has been saved
            if not os.path.exists(os.path.join(self.result_path['stage_1'], self.query_name + '.pkl')):
                result = self.stage_1(
                    query_faces, self.default_feature_folder)
                end = time.time()
                print("[+] Execution time of stage 1", end-start, " second")
                write_result_to_file(self.query_name, result, os.path.join(
                    self.result_path["stage_1"], self.query_name + ".txt"))
                write_result(self.query_name, result, os.path.join(
                    self.result_path["stage_1"], self.query_name + ".pkl"))

        if isStage2:
            print("\n==============================================================================")
            print("\n                       [+] Stage 2 of searching:\n")
            print(
                "==============================================================================")
            with open(os.path.join(self.result_path["stage_1"], self.query_name + ".pkl"), 'rb') as f:
                result = pickle.load(f)
            training_set = self.form_training_set(
                result[:100], thresh=0.6, rmBadFaces=self._PEsolvePnP)
            training_data_filename = os.listdir(os.path.join(
                self.training_data_folder, self.query_name))[0]
            with open(os.path.join(self.training_data_folder, self.query_name, training_data_filename), 'rb') as f:
                training_set = pickle.load(f)
            result = self.stage_2(query_faces, training_set)
            write_result_to_file(self.query_name, result, os.path.join(
                self.result_path["stage_2"], self.query_name + ".txt"))
            write_result(self.query_name, result, os.path.join(
                self.result_path["stage_2"], self.query_name + ".pkl"))
        
        if isStage3:

            print("\n==============================================================================")
            print("\n                       [+] Stage 3 of searching:\n")
            print(
                "==============================================================================")

            with open(os.path.join(self.result_path["stage_2"], self.query_name + ".pkl"), 'rb') as f:
                result = pickle.load(f)
            svm_training_set = self.form_SVM_training_set(
                result[:200], thresh=0.7, rmBadFaces=self._PEsolvePnP)
            result = self.stage_3(query_faces, svm_training_set)
            write_result_to_file(self.query_name, result, os.path.join(
                self.result_path["stage_3"], self.query_name + ".txt"))
            write_result(self.query_name, result, os.path.join(
                self.result_path["stage_3"], self.query_name + ".pkl"))


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    query_folder = "../data/raw_data/queries/"
    names = ["9104", "9115", "9116", "9119", "9124", "9138", "9143"]
    image_sticher = ImageSticher()
    search_eng = SearchEngine(visualize_tool)
    print("[+] Initialized searh engine")
    # name = names[0]
    for name in names:

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
        print("============================================================================\n\n")
        print()
        print("                       QUERY CHARACTER : %s\n\n" % (name.upper()))
        print(
            "============================================================================")
        imgs_v = [cv2.imread(q) for q in query]
        masks_v = [cv2.imread(m) for m in masks]
        search_eng.query_name = name
        visualize_tool.visualize_images(
            [imgs_v, masks_v], "Query : " + name, save_path="../result/" + name + "_query.jpg")       
        search_eng.searching(query, masks)

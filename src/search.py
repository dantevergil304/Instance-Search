from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras.engine import Model
from keras import backend as K
from feature_extraction import extract_feature_from_face, extract_database_faces_features
from apply_super_resolution import apply_super_res
from face_extraction_queries import detect_face_by_path
from vgg_finetune import fine_tune, extract_feature_from_face_list
from sticher import ImageSticher
from keras.models import load_model
from util import calculate_average_faces_sim, cosine_similarity, mean_max_similarity, write_result_to_file, write_result, create_stage_folder, adjust_size_different_images, create_image_label, max_mean_similarity, max_max_similarity
from scipy import stats
from PIL import Image
from checkGBFace_solvePnP import getFaceRotationAngles
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from natsort import natsorted
from checkGBFace import GoodFaceChecker

import numpy as np
import json
import pickle
import os
import cv2
import time
import sys
import glob
import multiprocessing
import subprocess


class SearchEngine(object):
    def __init__(self, image_sticher):
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

        self.result_path = os.path.abspath(
            os.path.join(self.cfg["result"], self.cfg["config"]))

        self.fine_tune_feature_folder = os.path.abspath(os.path.join(
            self.cfg["features"]["VGG_fine_tuned_features"], self.cfg["config"]))

        self.vgg_fine_tune_model_path = os.path.abspath(
            os.path.join(self.cfg["models"]["VGG_folder"]["VGG_fine_tuned_folder"], self.cfg["config"]))

        self.svm_model_path = os.path.abspath(
            os.path.join(self.cfg['models']["SVM_folder"], self.cfg["config"]))

        self.vgg_training_data_folder = os.path.abspath(
            os.path.join(self.cfg["training_data"]["VGG_data"], self.cfg["config"]))

        self.svm_training_data_folder = os.path.abspath(
            os.path.join(self.cfg["training_data"]["SVM_data"], self.cfg["config"]))

        # Set up config
        self.rmBF_method = self.search_cfg["rmBadFacesMethod"]
        self.rmBF_landmarks_params = self.search_cfg["rmBadFacesLandmarkBasedParams"]

        # Making directories if not exists
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.fine_tune_feature_folder, exist_ok=True)
        os.makedirs(self.vgg_fine_tune_model_path, exist_ok=True)
        os.makedirs(self.svm_model_path, exist_ok=True)
        os.makedirs(self.vgg_training_data_folder, exist_ok=True)
        os.makedirs(self.svm_training_data_folder, exist_ok=True)

        self.query_name = None
        self.sticher = image_sticher
        self.fine_tune_vgg = None
        self.svm_clf = None
        self.n_jobs = self.search_cfg['n_jobs']
        self.good_face_checker = GoodFaceChecker(
            method=self.rmBF_landmarks_params["check_face_pose_method"], checkBlur=(self.rmBF_landmarks_params["landmark_type"]=="True"))

    def remove_bad_faces(self, query):
        '''
        Parameters:
        - query: [(face matrix, feature vector)]
        Returns:
        - query_final: resulting query after remove some bad faces. (same format at parameter)
        '''
        n = len([q for q in query if q is not None])
        confs = [0] * 4
        query_final = []

        for i in range(n):
            if query[i] is not None:
                confs[i] = sum([cosine_similarity(query[i][1], query[j][1])
                                for j in range(n) if i != j and query[j] is not None])

        if n > 1:
            for i in range(4):
                if query[i] is not None:
                    mean = sum([confs[j] for j in range(n) if i !=
                                j and confs[j] != 0]) / (n - 1)
                    if confs[i] + 0.05 >= mean:
                        query_final.append(query[i])
                    else:
                        query_final.append(None)
                else:
                    query_final.append(None)
        else:
            return query

        if query_final.count(None) == 4:
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

            face = cv2.resize(face, (256, 256))

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
            rotation_vecs[idx] = np.minimum(
                180 - rotation_vecs[idx], rotation_vecs[idx])
            if abs(rotation_vecs[idx][0]) < 25 and abs(rotation_vecs[idx][1]) < 15:
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

        # filename = self.query_name + '_' + 'thresh=' + str(thresh)
        # if rmBadFaces is not None:
        #    filename += rmBadFaces.__name__

        # save_path = os.path.join(self.training_data_folder, self.query_name)
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)

        # self.sticher.process_training_set(training_set_path)

        return [X, Y]

    def form_training_set_using_best_face_in_each_shot(self, result, rmBadFaces=None):
        X, Y, landmarks_info = [], [], []

        for seqNum in range(50):  # Get positive samples
            record = result[seqNum]
            shot_id = record[0]
            video_id = shot_id.split('_')[0][4:]

            # Get best face
            best_face_data = sorted(calculate_average_faces_sim(
                record), key=lambda x: x[1])[-1]

            frame_file = best_face_data[0][0]
            frame = cv2.imread(os.path.join(
                self.frames_folder, 'video' + video_id, shot_id, frame_file))
            height, width = frame.shape[:2]

            # Get landmark of best face
            with open(os.path.join(self.faces_folder, 'video' + video_id, shot_id + '.pickle'), 'rb') as f:
                faces = pickle.load(f)
            with open(os.path.join(self.landmarks_folder, 'video' + video_id, shot_id + '.pickle'), 'rb') as f:
                landmarks = pickle.load(f)

            best_face_landmark = None
            for face, landmark in zip(faces, landmarks):
                if face == best_face_data[0]:
                    best_face_landmark = landmark
                    break
            image_points = []
            for i in range(int(len(best_face_landmark)/2.)):
                x, y = int(best_face_landmark[i]), int(best_face_landmark[i+5])
                image_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
            image_points = np.array(image_points, dtype='double')

            x, y, _x, _y = best_face_data[0][1]
            best_face = frame[y:_y, x:_x]
            X.append(best_face)
            landmarks_info.append((image_points, (height, width)))

            Y.append(1)

        for seqNum in range(50, 100):  # Get negative sample
            record = result[seqNum]
            shot_id = record[0]
            video_id = shot_id.split('_')[0][4:]

            best_face_data = sorted(calculate_average_faces_sim(
                record), key=lambda x: x[1])[-1]

            frame_file = best_face_data[0][0]
            frame = cv2.imread(os.path.join(
                self.frames_folder, 'video' + video_id, shot_id, frame_file))

            # Get landmark of best face
            with open(os.path.join(self.faces_folder, 'video' + video_id, shot_id + '.pickle'), 'rb') as f:
                faces = pickle.load(f)
            with open(os.path.join(self.landmarks_folder, 'video' + video_id, shot_id + '.pickle'), 'rb') as f:
                landmarks = pickle.load(f)

            best_face_landmark = None
            for face, landmark in zip(faces, landmarks):
                if face == best_face_data[0]:
                    best_face_landmark = landmark
                    break
            image_points = []
            for i in range(int(len(best_face_landmark)/2.)):
                x, y = int(best_face_landmark[i]), int(best_face_landmark[i+5])
                image_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
            image_points = np.array(image_points, dtype='double')

            x, y, _x, _y = best_face_data[0][1]
            best_face = frame[y:_y, x:_x]

            X.append(best_face)
            landmarks_info.append((image_points, (height, width)))

            Y.append(0)

        if rmBadFaces is not None:
            X, Y, _, _ = rmBadFaces(
                X, Y, landmarks_info)

        return [X, Y]

    def form_SVM_training_set(self, result, thresh=0.7, rmBadFaces=None):
        X, Y = self.form_training_set(result, thresh, rmBadFaces)

        print('[+] Extracting face features')
        model_path = os.path.join(
            self.vgg_fine_tune_model_path, self.query_name, 'vgg_model.h5')
        features = extract_feature_from_face_list(model_path, X)
        print('[+] Finished extracting face features')

        K.clear_session()
        return [features, Y], [X, Y]

    def uniprocess_stage_1(self, query, feature_folder, isStage3=False, block_interval=None):
        '''
        Parameters:
        - query: [(face matrix, feature vector)]
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
        print("[+] Current feature folder : %s\n" % (feature_folder))
        video_feature_files = \
            natsorted([(file, os.path.join(feature_folder, file))
                       for file in os.listdir(feature_folder)])
        # shot_feature_files = [(os.path.basename(path), path) for path in glob.iglob(
        #     feature_folder + '/**/*.pickle', recursive=True)]

        if block_interval:
            # start = block_id * self.search_cfg['blocksize']
            # end = start + self.search_cfg['blocksize']
            video_feature_files = \
                video_feature_files[block_interval[0]:block_interval[1]]

        cosine_similarity = []
        classification_score = []

        print('[+] Start to compute the similarity between person and each shot\n')
        idx = 0
        for video_feature_file in video_feature_files:
            video_id = video_feature_file[0].split('.')[0]
            print(f'Processing video {video_id}')
            with open(video_feature_file[1], 'rb') as f:
                video_feature = pickle.load(f)

            # for idx, shot_feature_file in enumerate(shot_feature_files):
            for shot_id, shot_faces_feat in video_feature.items():
                idx += 1
                # shot_id = shot_feature_file[0].split(".")[0]
                # video_id = shot_id.split('_')[0][4:]
                # print('[id: %d], computing similarity for %s' %
                #       (idx, shot_id))
                # print(len(result))
                # feature_path = shot_feature_file[1]
                face_path = os.path.join(
                    self.faces_folder, video_id, shot_id + '.pickle')
                # with open(feature_path, "rb") as f:
                #     shot_faces_feat = pickle.load(f)
                with open(face_path, "rb") as f:
                    shot_faces = pickle.load(f)
                # shot faces is a list with elements consist of  ((frame, (x1, y1, x2, y2)), face features)
                shot_faces = list(zip(shot_faces, shot_faces_feat))

                # print("\t%s , number of faces : %d" % (shot_id, len(shot_faces)))

                # shot_faces = shot_faces_feat
                sim, frames_with_bb_sim = mean_max_similarity(
                    query, shot_faces)

                if isStage3:
                    arr = [self.svm_clf.decision_function(
                        face_feat) for face_feat in shot_faces_feat]
                    decision_score = max(arr)
                    exact_distance = decision_score / \
                        np.linalg.norm(self.svm_clf.coef_)

                    cosine_similarity.append(sim)
                    classification_score.append(
                        np.expand_dims(exact_distance, 0))

                # Result is a list of elements consist of (shot_id, similarity(query, shot_id), corresponding matrix faces like explaination (1)
                result.append((shot_id, sim, frames_with_bb_sim))
            #     if len(result) == 100:
            #         break
            # if len(result) == 100:
            #     break

        print('[+] Finished computing similarity for all shots')

        if isStage3:
            person_similarity = 0.8 * stats.zscore(
                cosine_similarity) + 0.2 * stats.zscore(classification_score)
            shot_id, _, frames_with_bb_sim = zip(*result)
            result = list(zip(shot_id,
                              person_similarity, frames_with_bb_sim))

        result.sort(reverse=True, key=lambda x: x[1])
        print("[+] Search completed")
        with open(os.path.join('../temp', str(block_interval) + '.pkl'), 'wb') as f:
            pickle.dump(result, f)
        return result[:1000]

    def multiprocess_stage_1(self, query, feature_folder, isStage3=False):
        total_videos = len(os.listdir(self.frames_folder))
        avg_video_per_process = total_videos // self.n_jobs
        remain_videos = total_videos % self.n_jobs

        processes = []

        blocks = []
        start_idx = 0
        end_idx = 0
        for job_id in range(self.n_jobs):
            start_idx = end_idx

            batch_size = avg_video_per_process
            if job_id < remain_videos:
                batch_size += 1

            end_idx = start_idx + batch_size
            print('BLock Interval:', start_idx, end_idx)
            blocks.append((start_idx, end_idx))

        arg = [(query, feature_folder, isStage3, block) for block in blocks]
        with multiprocessing.get_context("spawn").Pool() as pool:
            result = pool.starmap(self.uniprocess_stage_1, arg)

        #     p = multiprocessing.Process(target=self.uniprocess_stage_1,
        #                                 args=(query, feature_folder,
        #                                       isStage3, (start_idx, end_idx)))
        #     p.daemon = True
        #     processes.append(p)
        #     p.start()

        # for process in processes:
        #     process.join()

    def stage_1(self, query, feature_folder, isStage3=False, multiprocess=False):
        if multiprocess:
            if os.path.isdir('../temp'):
                subprocess.call(['rm', '-rf', '../temp'])
            os.mkdir('../temp')
            self.multiprocess_stage_1(query, feature_folder, isStage3)
            result = []
            for result_path in glob.glob('../temp/*pkl'):
                with open(result_path, 'rb') as f:
                    result.extend(pickle.load(f))

            subprocess.call(['rm', '-rf', '../temp'])
            result.sort(reverse=True, key=lambda x: x[1])
            return result[:1000]
        return self.uniprocess_stage_1(query, feature_folder, isStage3)[:1000]

    def stage_2(self, query, training_set, multiprocess=False):
        '''
        Parameter:
        - query: [((face matrix, img query path, binary mask path), feature vector)]
        - training_set: a training set
        '''
        print("[+] Begin stage 2 of searching")

        model_path = os.path.join(
            self.vgg_fine_tune_model_path, self.query_name, 'vgg_model.h5')
        if not os.path.exists(model_path):
            self.fine_tune_vgg = fine_tune(
                training_set, model_name=self.query_name)
            print("[+] Finished fine tuned VGG Face model")
        else:
            self.fine_tune_vgg = load_model(model_path)
            print("[+] Load fine tuned VGG Face model")

        feature_extractor = Model(
            self.fine_tune_vgg.input, self.fine_tune_vgg.get_layer('fc6').output)

        fine_tune_feature_folder = os.path.join(
            self.fine_tune_feature_folder, self.query_name)

        if not os.path.exists(fine_tune_feature_folder):
            os.makedirs(fine_tune_feature_folder)
            print("[+] Begin extract feature using fine tuned model")
            extract_database_faces_features(
                feature_extractor, self.frames_folder, self.faces_folder, fine_tune_feature_folder)
            print("[+] Finished extract feature")
        query_faces = []

        for face in query:
            # faces_features store extractly like query_faces_sr except with addtional information, feature of query faces
            feature = extract_feature_from_face(feature_extractor, face[0])
            query_faces.append((face[0], feature))

        K.clear_session()
        return self.stage_1(query_faces, fine_tune_feature_folder, multiprocess=multiprocess)

    def stage_3(self, query, training_set=None, multiprocess=False):

        X, y = training_set[0], training_set[1]

        os.makedirs(os.path.join(self.svm_model_path,
                                 self.query_name), exist_ok=True)
        svm_model_path = os.path.join(
            self.svm_model_path, self.query_name, 'svm_model.pkl')

        if not os.path.exists(svm_model_path):
            print('[+] Begin Training SVM')
            self.svm_clf = SVC(probability=True, verbose=True,
                               random_state=42, kernel='linear', decision_function_shape='ovo')
            self.svm_clf.fit(X, y)
            print('[+] Fininshed Training SVM')

            with open(svm_model_path, 'wb') as f:
                pickle.dump(self.svm_clf, f)
        else:
            print('[+] SVM model already exists')
            with open(svm_model_path, 'rb') as f:
                self.svm_clf = pickle.load(f)

        query_faces = []
        vgg_fine_tune_model_path = os.path.join(
            self.vgg_fine_tune_model_path, self.query_name, 'vgg_model.h5')
        fine_tune_vgg = load_model(vgg_fine_tune_model_path)
        feature_extractor = Model(
            fine_tune_vgg.input, fine_tune_vgg.get_layer('fc6').output)
        for face in query:
            # faces_features store extractly like query_faces_sr except with addtional information, feature of query faces
            feature = extract_feature_from_face(feature_extractor, face[0])
            query_faces.append((face[0], feature))

        K.clear_session()

        fine_tune_feature_folder = os.path.join(
            self.fine_tune_feature_folder, self.query_name)

        return self.stage_1(query_faces, fine_tune_feature_folder, isStage3=True, multiprocess=multiprocess)

    def searching(self, query, mask, isStage1=True, isStage2=False, isStage3=False, multiprocess=False):

        root_result_folder = os.path.join(self.result_path, self.query_name)
        os.makedirs(root_result_folder, exist_ok=True)
        stage_1_execution_time = 0
        stage_2_execution_time = 0
        stage_3_execution_time = 0

        # Detect faces in query
        query_faces, bb, landmarks = detect_face_by_path(query, mask)
        K.clear_session()
        print("[+] Detected faces from query")

        # Convert landmark from MTCNN format to list of landmark points
        landmark_list = []
        for landmark in landmarks:
            landmark_points = []
            for i in range(int(len(landmark)/2.)):
                    x, y = int(landmark[i]), int(landmark[i+5])

                    landmark_points.append((x, y))
            landmark_list.append(np.array(landmark_points, dtype='double'))


        faces_v = list(zip(*query_faces))[0]
        v_faces = adjust_size_different_images(faces_v, 341, 341/2)

        # Get size of each frame in query
        frames_size = []
        for qpath in query:
            frame = cv2.imread(qpath)
            frames_size.append(frame.shape[:2])

        # Apply super resolution
        super_res_faces_path = os.path.join(
            root_result_folder, 'super_res_faces.pkl')
        faces_sr = []
        if not os.path.exists(super_res_faces_path):
            faces_sr = []
            for idx, face in enumerate(faces_v):
                if face is not None:
                    faces_sr.append(apply_super_res(face))
                else:
                    faces_sr.append(None)
            with open(super_res_faces_path, 'wb') as f:
                pickle.dump(faces_sr, f)
            print("[+] Applied Super Resolution to detected faces")
        else:
            with open(super_res_faces_path, 'rb') as f:
                faces_sr = pickle.load(f)
            print("[+] Super resolution faces already existed!")
        K.clear_session()

        v_faces_sr = adjust_size_different_images(faces_sr, 341, 341)

        temp_1 = []
        temp_2 = []
        for i, face in enumerate(v_faces):
            if face is None:
                temp_1.append(np.zeros((341, 192, 3), dtype=np.uint8))
                temp_2.append(np.zeros((341, 192, 3), dtype=np.uint8))
            else:
                temp_1.append(face)
                temp_2.append(v_faces_sr[i])

        imgs_v = [cv2.imread(q) for q in query]
        for i, img in enumerate(imgs_v):
            if bb[i]:
                cv2.rectangle(img, (bb[i][0], bb[i][1]),
                              (bb[i][2], bb[i][3]), (0, 255, 0), 5)

        # Extract query feature
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer(self.search_cfg["feature_descriptor"]).output
        default_vgg = Model(vgg_model.input, out)

        faces_features = []
        for face in faces_sr:
            if face is not None:
                feature = extract_feature_from_face(default_vgg, face)
                faces_features.append((face, feature))
            else:
                faces_features.append(None)

        K.clear_session()
        print("[+] Extracted feature of query images")

        # Remove Bad Faces in query
        if self.rmBF_method == 'peking':
            query_faces = self.remove_bad_faces(faces_features)
        elif self.rmBF_method == 'landmark_based':
            if self.rmBF_landmarks_params['landmark_type'] == 'dlib':
                query_faces = faces_features
                for idx, (face, frame_size) in enumerate(zip(faces_v, frames_size)):
                    if face is not None:
                        if not self.good_face_checker.isGoodFace(face, frame_size):
                            query_faces[idx] = None
            elif self.rmBF_landmarks_params['landmark_type'] == 'mtcnn':
                query_faces = faces_features
                for idx, (face, frame_size, landmark) in enumerate(zip(faces_v, frames_size, landmark_list)):
                    if face is not None:
                        if not self.good_face_checker.isGoodFace(face, frame_size, landmark):
                            query_faces[idx] = None
            

        # Visulize the query after remove bad faces
        temp = []
        for query_face in query_faces:
            if not query_face:
                temp.append(np.zeros((341, 192, 3), dtype=np.uint8))
            else:
                temp.append(query_face[0])
        temp = adjust_size_different_images(temp, 341, 341)

        frames_faces_label = create_image_label(
            "Detect faces in frames", (192, 350, 3))
        imgs_v = [cv2.resize(img, (341, 192)) for img in imgs_v]
        imgs_v = [frames_faces_label] + imgs_v

        before_sr_label = create_image_label(
            "Before apply SR", (temp_1[0].shape[0], 350, 3))
        temp_1 = [before_sr_label] + temp_1

        after_sr_label = create_image_label(
            "After apply SR", (temp_2[0].shape[0], 350, 3))
        temp_2 = [after_sr_label] + temp_2

        rmbf_label = create_image_label(
            "After remove bad faces", (temp[0].shape[0], 350, 3))
        temp = [rmbf_label] + temp

        self.sticher.stich(matrix_images=[imgs_v, temp_1, temp_2,  temp], title="Preprocess query",
                           save_path=os.path.join(root_result_folder, "preprocess.jpg"), size=None, reduce_size=True)

        query_faces = [query_face for query_face in query_faces if query_face]

        if isStage1:
            print(
                "\n==============================================================================")
            print("\n                       [+] Stage 1 of searching:\n")
            print(
                "==============================================================================")
            stage_1_path = os.path.join(root_result_folder, "stage 1")
            create_stage_folder(stage_1_path)
            start = time.time()
            default_feature_folder = os.path.join(
                self.default_feature_folder, self.search_cfg["feature_descriptor"])
            result = self.stage_1(
                query_faces, default_feature_folder, multiprocess=multiprocess)
            stage_1_execution_time = time.time() - start

            write_result_to_file(self.query_name, result, os.path.join(
                root_result_folder, 'stage 1', "result.txt"))
            write_result(self.query_name, result, os.path.join(
                root_result_folder, "stage_1.pkl"))
            # self.sticher.save_shots_max_images(
            #     result, os.path.join(stage_1_path))

        if isStage2:
            print(
                "\n==============================================================================")
            print("\n                       [+] Stage 2 of searching:\n")
            print(
                "==============================================================================")
            stage_2_path = os.path.join(root_result_folder, "stage 2")
            create_stage_folder(stage_2_path)

            start = time.time()
            stage_1_result_file = os.path.join(
                root_result_folder, "stage_1.pkl")
            with open(stage_1_result_file, 'rb') as f:
                result = pickle.load(f)

            if not os.path.isdir(os.path.join(self.vgg_training_data_folder, self.query_name)):
                os.mkdir(os.path.join(
                    self.vgg_training_data_folder, self.query_name))

            training_set_path = os.path.join(
                self.vgg_training_data_folder, self.query_name, "training_data.pkl")

            if not os.path.exists(training_set_path):
                # training_set = self.form_training_set(
                #     result[:100], thresh=0.8, rmBadFaces=self._PEsolvePnP)
                training_set = self.form_training_set_using_best_face_in_each_shot(
                    result[:100], rmBadFaces=self._PEsolvePnP)
                with open(training_set_path, "wb") as f:
                    pickle.dump(training_set, f)
                self.sticher.process_training_set(
                    training_set_path, save_path=os.path.join(self.vgg_training_data_folder, self.query_name))
                print("[+] Builded training data")
            else:
                with open(training_set_path, 'rb') as f:
                    training_set = pickle.load(f)
                print("[+] Loaded training data")

            result = self.stage_2(query_faces, training_set,
                                  multiprocess=multiprocess)
            stage_2_execution_time = time.time() - start

            write_result_to_file(self.query_name, result, os.path.join(
                root_result_folder, 'stage 2', "result.txt"))
            write_result(self.query_name, result, os.path.join(
                root_result_folder, "stage_2.pkl"))
            # self.sticher.save_shots_max_images(
            #     result, stage_2_path)

        if isStage3:
            print(
                "\n==============================================================================")
            print("\n                       [+] Stage 3 of searching:\n")
            print(
                "==============================================================================")
            stage_3_path = os.path.join(root_result_folder, "stage 3")
            create_stage_folder(stage_3_path)
            start = time.time()

            stage_2_result_file = os.path.join(
                root_result_folder, "stage_2.pkl")
            with open(stage_2_result_file, 'rb') as f:
                result = pickle.load(f)

            if not os.path.isdir(os.path.join(self.svm_training_data_folder, self.query_name)):
                os.mkdir(os.path.join(
                    self.svm_training_data_folder, self.query_name))

            training_set_path = os.path.join(
                self.svm_training_data_folder, self.query_name, "training_data.pkl")
            faces_training_set_path = os.path.join(
                self.svm_training_data_folder, self.query_name, "faces_training_data.pkl")

            if not os.path.exists(training_set_path):
                training_set, faces_training_set = self.form_SVM_training_set(
                    result[:200], thresh=0.65, rmBadFaces=self._PEsolvePnP)

                with open(training_set_path, "wb") as f:
                    pickle.dump(training_set, f)

                with open(faces_training_set_path, 'wb') as f:
                    pickle.dump(faces_training_set, f)
                self.sticher.process_training_set(
                    faces_training_set_path, save_path=os.path.join(self.svm_training_data_folder, self.query_name))

                print("[+] Builded training data")
            else:
                print('Training data already exists')
                with open(training_set_path, 'rb') as f:
                    training_set = pickle.load(f)
                print("[+] Loaded training data")

            result = self.stage_3(query_faces, training_set,
                                  multiprocess=multiprocess)
            stage_3_execution_time = time.time() - start

            write_result_to_file(self.query_name, result, os.path.join(
                root_result_folder, 'stage 3', "result.txt"))
            write_result(self.query_name, result, os.path.join(
                root_result_folder, "stage_3.pkl"))
            # self.sticher.save_shots_max_images(
            #     result, stage_3_path)

        with open(os.path.join(self.result_path, self.query_name, 'log.txt'), 'w') as f:
            f.write("Execution time of stage 1 : " +
                    str(stage_1_execution_time))
            f.write("\nExecution time of stage 2 : " +
                    str(stage_2_execution_time))
            f.write("\nExecution time of stage 3 : " +
                    str(stage_3_execution_time))


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    query_folder = "../data/raw_data/queries/"
    # names = ["9104", "9115", "9116", "9119", "9124", "9138", "9143"]
    names = ["chelsea", "darrin", "garry", "heather",
             "jack", "jane", "max", "minty", "mo", "zainab"]
    # names = ["chelsea", "garry", "heather",
    #          "jack", "jane", "minty", "mo", "zainab"]
    # names = ['chelsea']
    search_engine = SearchEngine(ImageSticher())
    print("[+] Initialized searh engine")
    for name in names:
        query = [
            name + ".1.src.png",
            name + ".2.src.png",
            name + ".3.src.png",
            name + ".4.src.png"

        ]
        masks = [
            name + ".1.mask.png",
            name + ".2.mask.png",
            name + ".3.mask.png",
            name + ".4.mask.png"
        ]

        query = [os.path.join(query_folder, q) for q in query]
        masks = [os.path.join(query_folder, m) for m in masks]
        print("============================================================================\n\n")
        print()
        print("                       QUERY CHARACTER : %s\n\n" % (name))
        print(
            "============================================================================")
        imgs_v = [cv2.imread(q) for q in query]
        masks_v = [cv2.imread(m) for m in masks]
        search_engine.query_name = name

        search_engine.sticher.stich(matrix_images=[imgs_v, masks_v], title="Query : " + name,
                                    save_path=os.path.join(search_engine.result_path, name, "query.jpg"))

        search_engine.searching(
            query, masks, isStage1=True, isStage2=False, isStage3=False, multiprocess=True)

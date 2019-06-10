import numpy as np
import cv2
import os
import json
import random
import pickle
import math
from util import calculate_average_faces_sim, cosine_similarity, mean_max_similarity


class ImageSticher():
    def __init__(self):
        with open("../cfg/config.json", "r") as f:
            self.cfg = json.load(f)
        with open("../cfg/search_config.json", "r") as f:
            self.search_cfg = json.load(f)

        self.faces_folder = os.path.abspath(
            self.cfg["processed_data"]["faces_folder"])
        self.frames_folder = os.path.abspath(
            self.cfg["processed_data"]["frames_folder"])

    def stich(self, matrix_images, title, save_path=None, size=(341, 192), reduce_size=False):
        imgs = []
        for i, row in enumerate(matrix_images):
            if size is None:
                h = np.hstack(tuple(row))
            else:
                h = np.hstack(tuple([cv2.resize(img, size) for img in row]))
            imgs.append(h)
        img = np.vstack(tuple(imgs))
        if reduce_size:
            img = cv2.resize(
                img, (int(img.shape[0]/1.5), int(img.shape[1]/1.5)))
        if save_path:
            print("Save %s image to %s" % (title, save_path))
            cv2.imwrite(save_path, img)
    
    def save_query_shot_face(self, query_shot_feat, good_query_shot_feat, save_path):
        result_img = None
        result_img_width = 1000
        face_per_row = 10

        faces_row = None 
        good_faces_row = None
        default_height = 100
        for idx, (face_feat, good_face_feat) in enumerate(zip(query_shot_feat, good_query_shot_feat)):
            face = face_feat[0]
            gface = None if good_face_feat is None else good_face_feat[0]

            height, width = face.shape[:2]  
            new_width = int(default_height * width / height)

            # Process face
            resized_face = cv2.resize(face, (new_width, default_height))
            if faces_row is None:
                faces_row = resized_face
            else:
                faces_row = np.hstack((faces_row, resized_face))

            # Process good face
            if gface is None:
                resized_face = np.zeros_like(resized_face, np.uint8)
            if good_faces_row is None:
                good_faces_row = resized_face
            else:
                good_faces_row = np.hstack((good_faces_row, resized_face))
                

            if (idx + 1) % face_per_row == 0:
                hrow, wrow = good_faces_row.shape[:2]
                new_hrow = int(result_img_width * hrow / wrow)

                faces_row = cv2.resize(faces_row, (result_img_width, new_hrow))
                good_faces_row = cv2.resize(good_faces_row, (result_img_width, new_hrow))

                if result_img is None:
                    result_img = np.vstack((faces_row, good_faces_row))
                else:
                    result_img = np.vstack((result_img, np.zeros((50, result_img_width, 3))))
                    result_img = np.vstack((result_img, faces_row, good_faces_row))

                faces_row = None
                good_faces_row = None

        if faces_row is not None:
            if idx % face_per_row != 0:
                num_face_empty = face_per_row - (idx % face_per_row) 
                approximate_width_per_face = faces_row.shape[1] // (idx % face_per_row)

                faces_row = np.hstack((faces_row, np.zeros((faces_row.shape[0], approximate_width_per_face * num_face_empty, 3))))
                good_faces_row = np.hstack((good_faces_row, np.zeros((good_faces_row.shape[0], approximate_width_per_face * num_face_empty, 3))))

            hrow, wrow = good_faces_row.shape[:2]
            new_hrow = int(result_img_width * hrow / wrow)

            faces_row = cv2.resize(faces_row, (result_img_width, new_hrow))
            good_faces_row = cv2.resize(good_faces_row, (result_img_width, new_hrow))

            if result_img is None:
                result_img = np.vstack((faces_row, good_faces_row))
            else:
                result_img = np.vstack((result_img, np.zeros((50, result_img_width, 3))))
                result_img = np.vstack((result_img, faces_row, good_faces_row))

        cv2.imwrite(save_path, result_img)

    def save_shots_max_images(self, result, save_folder):
        for i, record in enumerate(result):
            shot_id = record[0]
            frames = calculate_average_faces_sim(record)
            frames = sorted(frames, reverse=True, key=lambda x: x[1])
            frame = frames[0]
            name = frame[0][0]
            img_path = os.path.join(
                self.frames_folder, "video" + shot_id.split('_')[0][4:], shot_id, name)
            x1, y1, x2, y2 = frame[0][1]
            img = cv2.imread(img_path)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            file_name = str(i+1) + "_" + \
                shot_id + "_" + str(record[1][0][0]) + ".jpg"
            save_path = os.path.join(save_folder, file_name)
            self.stich([[img]], shot_id, save_path=save_path)

    def process_result(self, result, save_folder):
        print("[+] Visualize results")
        for i, record in enumerate(result):
            shot_id = record[0]
            frames = calculate_average_faces_sim(record)
            n = 6 if len(frames) > 6 else len(frames)
            frames = sorted(frames, reverse=True, key=lambda x: x[1])
            frames = frames[:n]
            frames_with_faces = []
            matrix = []
            for index, frame in enumerate(frames):
                name = frame[0][0]
                img_path = os.path.join(
                    self.frames_folder, "video" + shot_id.split('_')[0][4:], shot_id, name)
                print(img_path)
                img = cv2.imread(img_path)
                x1, y1, x2, y2 = frame[0][1]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frames_with_faces.append(img)
                if len(frames_with_faces) == 3:
                    matrix.append(frames_with_faces)
                    frames_with_faces = []
            m = len(frames_with_faces)
            if m > 0:
                if m < 3:
                    for i in range(3 - m):
                        frames_with_faces.append(
                            np.zeros((341, 192, 3), dtype=np.uint8))
                matrix.append(frames_with_faces)

            file_name = str(i+1) + "_" + \
                shot_id + "_" + str(record[1][0]) + ".jpg"
            save_path = os.path.join(save_folder, file_name)
            self.stich(matrix, shot_id, save_path=save_path, size=(511, 288))

    def process_training_set(self, training_set_path, shape=(40, 10), save_path=None):
        with open(training_set_path, "rb") as f:
            training_set = pickle.load(f)
        training_set = list(zip(training_set[0], training_set[1]))
        file_name = training_set_path.split("/")[-1].replace(".pkl", "")
        query_id = file_name[:4]
        print("[+] Loaded dataset ", file_name)
        print("[+] Training set size : ", len(training_set))

        no_samples = len(training_set)
        if shape is None:
            size = round(math.sqrt(no_samples))
            shape = (size, size)

        rows = []
        row = []
        num_img = 0
        for count, sample in enumerate(training_set):
            face = cv2.resize(sample[0], (100, 100))
            text = str(sample[1])
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = np.zeros(face.shape)
            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary
            textX = (face.shape[1] - textsize[0]) / 2
            textY = (face.shape[0] + textsize[1]) / 2

            # add text centered on image
            cv2.putText(label, text, (int(textX), int(textY)),
                        font, 1, (255, 255, 255), 2)
            sample_img = np.hstack((face, label))
            row.append(sample_img)

            if len(row) >= shape[1] or count == no_samples - 1:
                # print("Length row : ", len(row))

                if len(row) < shape[1]:
                    for _ in range(shape[1] - len(row)):
                        row.append(
                            np.zeros((face.shape[0], face.shape[0] * 2, 3)))

                rows.append(np.hstack(tuple(row)))
                if len(rows) > shape[0] or count == no_samples - 1:
                    img = np.vstack(tuple(rows))
                    cv2.imwrite(os.path.join(
                        save_path, 'visualize_training_data_' + str(num_img) + '.jpg'), img)
                    num_img += 1
                    rows = []
                    if count == no_samples-1:
                        break
                row = []


if __name__ == '__main__':
    training_set_path = "../data/training_data/9143_PEsolvePnP_dataset.pkl"
    tools = ImageSticher()
    tools.process_training_set(training_set_path)

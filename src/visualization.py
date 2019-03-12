import numpy as np
import cv2
import os
import json
import random
import pickle
import math
from util import calculate_average_faces_sim, cosine_similarity, mean_max_similarity


class VisualizeTools():
    def __init__(self):
        with open("../cfg/config.json", "r") as f:
            self.cfg = json.load(f)
        with open("../cfg/search_config.json", "r") as f:
            self.search_cfg = json.load(f)

        self.faces_folder = os.path.abspath(
            self.cfg["processed_data"]["faces_folder"])
        self.frames_folder = os.path.abspath(
            self.cfg["processed_data"]["frames_folder"])

    def visualize_images(self, matrix_images, title, save_path=None, size=(341, 192)):
        imgs = []
        for row in matrix_images:
            h = np.hstack(tuple([cv2.resize(img, size) for img in row]))
            imgs.append(h)
        img = np.vstack(tuple(imgs))
        if save_path:
            cv2.imwrite(save_path, img)
        #cv2.imshow(title, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def view_result(self, result, save_folder, query_name):
        print("[+] Visualize results")
        for i, record in enumerate(result):
            shot_id = record[0]
            print("\tTop %d , Shot %s, Similarity %f" %
                  (i + 1, shot_id, record[1]))
            frames = calculate_average_faces_sim(record)
            n = 5 if len(frames) > 5 else len(frames)
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
            save_path = os.path.join(save_folder, query_name, file_name)
            self.visualize_images([frames_with_faces],
                                  shot_id, save_path=save_path)

    def view_training_set(self, training_set_path, shape=(40, 40)):
        with open(training_set_path, "rb") as f:
            training_set = pickle.load(f)
        training_set = list(zip(training_set[0], training_set[1]))
        file_name = training_set_path.split("/")[-1].replace(".pkl", "")
        query_id = file_name[:4]
        print("[+] Loaded dataset ", file_name)
        print("[+]Training set size : ", len(training_set))

        no_samples = len(training_set)
        if shape is None:
            size = round(math.sqrt(no_samples))
            shape = (size, size)

        if not os.path.isdir(os.path.join(self.cfg['training_data']['visualized_data'], query_id)):
            os.mkdir(os.path.join(
                self.cfg['training_data']['visualized_data'], query_id))

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
                        self.cfg["training_data"]["visualized_data"], query_id, file_name + '_' + str(num_img) + ".jpg"), img)
                    num_img += 1
                    rows = []
                    if count == no_samples-1:
                        break
                row = []


if __name__ == '__main__':
    training_set_path = "../data/training_data/9143_PEsolvePnP_dataset.pkl"
    tools = VisualizeTools()
    tools.view_training_set(training_set_path)

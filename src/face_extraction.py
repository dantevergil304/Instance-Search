import tensorflow as tf
import numpy as np
import pickle
import json
import cv2
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join('..', '3rd_party')))
from ServiceMTCNN import detect_face as lib


def extract_faces_from_frames_folder(input_frames_folder, output_faces_folder, output_landmarks_folder):
    '''
    Extract faces for all shots in input_frames_folder. Each shot
    will be saved on disk and represented by a list whose element
    is a tuple (frame_name, (x1, y1, x2, y2)).

    Parameter:
    - input_frames_folder: path to folder of shot folders, each shot folder
    contains many frames.
    - output_faces_folder: path for saving face
    '''
    begin = time.time()
    sess = tf.Session()
    pnet, rnet, onet = lib.create_mtcnn(sess, None)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    print("[+] Initialized MTCNN modules")

    print("[+] Accessed frames folder : %s" % (input_frames_folder))
    num_of_new_files = 0
    frames_folders = [(folder, os.path.join(input_frames_folder, folder))
                      for folder in os.listdir(input_frames_folder)]
    print(frames_folders)
    for i, frames_folder in enumerate(frames_folders):
        # Check if there is no free space on hard disk
        statvfs = os.statvfs('/')
        if statvfs.f_frsize * statvfs.f_bavail / (10**6 * 1024) < 5:
            print(
                '\033[93mWarning: Stop process. There is no free space left!\033[0m')
            break

        # MAIN
        shot_id, frames_folder_path = frames_folder[0], frames_folder[1]
        face_save_path = os.path.join(output_faces_folder, shot_id + ".pickle")
        landmark_save_path = os.path.join(
            output_landmarks_folder, shot_id + ".pickle")
        print("\t\t[+] id : %d | Accessed frames folder of %s path : %s" %
              (i, shot_id, frames_folder_path))

        if os.path.exists(face_save_path):
            print("\t\t" + shot_id + '.pickle has already existed')
            continue

        num_of_new_files += 1
        faces_bbs = list()
        faces_landmarks = list()
        frames = [(file, os.path.join(frames_folder_path, file))
                  for file in os.listdir(frames_folder_path)]

        for frame in frames:
            frame_id, frame_path = frame[0], frame[1]
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("\t\t[+] Load frame %s" % (frame_id))
            boxes, landmarks = lib.detect_face(
                img, minsize, pnet, rnet, onet, threshold, factor)
            if len(boxes) > 0:
                print("\t\t[+] Detected faces in frame %s" % (frame_id))
            else:
                print("\t\t[-] No faces were detected in frame")

            for index, box in enumerate(boxes):
                if box[4] < 0.9:  # remove faces with low confidence
                    continue
                x1, y1, x2, y2 = int(box[0]), int(
                    box[1]), int(box[2]), int(box[3])
                if y1 < 0:
                    y1 = 0
                if x1 < 0:
                    x1 = 0
                if y2 > img.shape[0]:
                    y2 = img.shape[0]
                if x2 > img.shape[1]:
                    x2 = img.shape[1]
                print("\t\tbox : ", box)
                face = (frame_id, (x1, y1, x2, y2))
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                #cv2.imshow(frame_id, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                faces_bbs.append(face)

            for landmark in np.transpose(landmarks):
                faces_landmarks.append(landmark)

        if len(faces_bbs) > 0:
            with open(face_save_path, 'wb') as f:
                pickle.dump(faces_bbs, f)
            with open(landmark_save_path, 'wb') as f:
                pickle.dump(faces_landmarks, f)
            print(faces_bbs)

            print("\t\t\033[93m[+] Saved faces to %s" % (face_save_path))
            print("\t\t[+] Saved landmarks to %s\033[0m" %
                  (landmark_save_path))

    end = time.time()
    elapsed_time = end - begin
    print("\tNumber of new file : ", num_of_new_files)
    print("\tTotal elapsed time: %f minutes and %d seconds" %
          (elapsed_time / 60, elapsed_time % 60))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    print("[+] Load config file")
    with open("../cfg/config.json", "r") as f:
        config = json.load(f)

    frames_folder = os.path.abspath(
        config["processed_data"]["5fps_png_frames_folder"])
    faces_folder = os.path.abspath(
        config["processed_data"]["5fps_png_faces_folder"])
    landmarks_folder = os.path.abspath(
        config["processed_data"]["5fps_png_landmarks_folder"])

    for vidId in range(0, 244):
        curVidFrameDir = os.path.join(frames_folder, 'video' + str(vidId))
        curVidFaceDir = os.path.join(faces_folder, 'video' + str(vidId))
        curVidLandmarkDir = os.path.join(
            landmarks_folder, 'video' + str(vidId))

        print('Video ID:', vidId)
        #print('Frames Directory:', curVidFrameDir)
        #print('Faces Directory:', curVidFaceDir)
        #print('Landmarks Directory:', curVidFrameDir)

        extract_faces_from_frames_folder(
            curVidFrameDir, curVidFaceDir, curVidLandmarkDir)

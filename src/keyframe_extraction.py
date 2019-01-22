import cv2
import os
import errno
import glob
import json


def getTotalFrame(path):
    # calculate the total frames of a video
    cap = cv2.VideoCapture(path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
    cnt += 1
    return cnt


def getDuration(cap):
    # duration = total_frame / FPS
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)


def KeyframeExtraction(input_path, output_path, sampling_rate=None):
    # path: path to video file
    # sampling_rate: the number of frames per second
    cap = cv2.VideoCapture(input_path)

    # total_frame = getTotalFrame(input_path)
    # duration = getDuration(cap)
    # fps = total_frame / duration
    if sampling_rate is not None:
        # coef = round(fps / sampling_rate)
        coef = round(25 / sampling_rate)

    filename = input_path.split('/')[-1]
    directory_path = os.path.join(output_path, filename.split('.')[0])
    print('...Extracting frames to ' + directory_path)
    try:
        os.makedirs(directory_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    index = 0
    label = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            if (sampling_rate is None) or (index % coef == 0):
                cv2.imwrite(os.path.join(directory_path, str(
                    label).zfill(5) + '.jpg'), frame)
                label += 1
        else:
            break
        index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with open("../cfg/config.json", "r") as f:
        config = json.load(f)

    raw_shot_folder = os.path.abspath(config["raw_data"]["raw_shots_folder"])
    frames_folder = os.path.abspath(config["processed_data"]["frames_folder"])

    for shot in glob.glob(os.path.join(raw_shot_folder, '*.mp4')):
        save_dir = os.path.join(
            frames_folder, os.path.basename(shot).split('.')[0])
        if not os.path.isdir(save_dir):
            KeyframeExtraction(shot, frames_folder, 5)

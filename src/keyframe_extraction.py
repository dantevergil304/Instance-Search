import cv2
import os
import errno
import glob
import json
import time


def getTotalFrame(cap):
    return cap.get(cv2.CAP_PROP_FRAME_COUNT)


def getDuration(cap):
    # duration = total_frame / FPS
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)


def KeyframeExtraction(input_path, output_path, sampling_rate=None, max_frame_per_shot=None):
    # path: path to video file
    # sampling_rate: the number of frames per second
    cap = cv2.VideoCapture(input_path)
    origin_fps = cap.get(cv2.CAP_PROP_FPS)
    print('Total Frame:', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame_per_shot is not None:
        sampling_rate = max_frame_per_shot * cap.get(cv2.CAP_PROP_FPS) / cap.get(cv2.CAP_PROP_FRAME_COUNT)

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
                # cv2.imwrite(os.path.join(directory_path, str(
                #     label).zfill(5) + '.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(os.path.join(directory_path, str(
                    label).zfill(5) + '.png'), frame)
                label += 1
        else:
            break
        index += 1

    cap.release()
    cv2.destroyAllWindows()

def extract_kf_shot_queries(shot_query_path, shot_query_frames_path, sampling_rate=None):
    for query_dir in glob.glob(os.path.join(shot_query_path, '*')):
        query_name = os.path.basename(query_dir)
        save_path = os.path.join(shot_query_frames_path, query_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for shot_video in glob.glob(os.path.join(query_dir, '*mp4')):
            KeyframeExtraction(shot_video, save_path, max_frame_per_shot=10, sampling_rate=sampling_rate)

if __name__ == '__main__':
    #####################################EXTRACT KF SHOT QUERY#####################################
    shot_query_path = '/storageStudents/K2015/duyld/hieudvm/Instance_Search/data/raw_data/queries/2018/tv18.person.example.shots'
    shot_query_frames_path = '/storageStudents/K2015/duyld/hieudvm/Instance_Search/data/raw_data/queries/2018/shot_query_frames/10_frame_per_shot'
    extract_kf_shot_queries(shot_query_path, shot_query_frames_path)
    

    #################################EXTRACT KF SHOT VIDEO DATABASE#################################
    # with open("../cfg/config.json", "r") as f:
    #     config = json.load(f)

    # raw_shot_folder = os.path.abspath(config["raw_data"]["raw_shots_folder"])
    # # frames_folder = os.path.abspath(config["processed_data"]["frames_folder"])
    # frames_folder = '/storageStudents/K2015/duyld/hieudvm/TestScript/folder_video_226/'

    # total_time = 0
    # extracted_shot = 0
    # for shot in glob.glob(os.path.join(raw_shot_folder, '*.mp4')):
    #     video_id = os.path.basename(shot).split('_')[0][4:]
    #     if video_id != '226':
    #         continue

    #     # Check if there is no free space on hard disk
    #     statvfs = os.statvfs('/')
    #     if statvfs.f_frsize * statvfs.f_bavail / (10**6 * 1024) < 5:
    #         print(
    #             '\033[93mWarning: Stop process. There is no free space left!\033[0m')
    #         break

    #     save_dir = os.path.join(
    #         frames_folder, os.path.basename(shot).split('.')[0])
    #     if not os.path.isdir(save_dir):
    #         begin = time.time()
    #         KeyframeExtraction(shot, frames_folder, 5)
    #         end = time.time()

    #         total_time += (end - begin)

    #         extracted_shot += 1
    #     print('[+] Number of extracted shots: %d' % (extracted_shot))

    # print('Total Elapsed Time: %f minutes and %d seconds' % (
    #     total_time/60, total_time % 60))

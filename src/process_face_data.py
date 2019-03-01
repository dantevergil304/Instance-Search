import cv2
import glob
import os
import pickle
import numpy as np
import time
import h5py
import sys

from keras_vggface import utils


def BatchGenerator(VIDEO_ID, batch_size=2000):
    x = []
    num_faces = 0
    shots_order_and_info = []
    num_files = 0
    generate_batch_t = time.time()

    for file in glob.glob('../data/processed_data/organized_faces/video' + str(VIDEO_ID) + '/*pickle'):
        num_files += 1
        # print('Processing %d shot(s)' % (num_files))

        with open(file, 'rb') as f:
            bbs = pickle.load(f)

        shot_id = os.path.basename(file).split('.')[0]
        shots_order_and_info.append((shot_id, len(bbs)))

        for frame_id, bb in bbs:
            frame_path = os.path.join(
                '../data/processed_data/frames/', shot_id, frame_id)
            frame = cv2.imread(frame_path)

            x1, y1, x2, y2 = bb

            face = frame[y1:y2, x1:x2]

            face = cv2.resize(face, (224, 224))

            face = face.astype(np.float32)

            face = utils.preprocess_input(face, version=1)

            x.append(face)
            num_faces += 1

            if num_faces == batch_size:
                x = np.array(x).reshape((-1, 224, 224, 3))
                print('Generate batch time: %d seconds' %
                      (time.time() - generate_batch_t))
                yield x
                x = []
                num_faces = 0
                generate_batch_t = time.time()

    if x != []:
        x = np.array(x).reshape((-1, 224, 224, 3))
        print('Generate batch time: %d seconds' %
              (time.time() - generate_batch_t))
        yield x

    yield shots_order_and_info


def SaveFaceBatchAndInfo(video_id):
    for idx, batch in enumerate(BatchGenerator(video_id)):
        if isinstance(batch, np.ndarray):
            print('Saving batch %d' % (idx))
            np.save('../data/processed_data/extracted_faces/video' + str(video_id) + '/batch' +
                    str(idx) + '.npy', batch)
            # with h5py.File('/gdrive/My Drive/Instance_Search/data/processed_data/extracted_faces/video0/gzip_batch' + str(idx) + '.hdf5', 'w') as f:
            #   batch = f.create_dataset('batch' + str(idx), data=batch, compression='gzip', compression_opts=9)
            # with open('/gdrive/My Drive/Instance_Search/data/processed_data/extracted_faces/video0/batch' + str(idx) + '.pkl', 'wb') as f:
            #   pickle.dump(batch, f)
        else:
            print('Saving info file')
            with open('../data/processed_data/extracted_faces/video' + str(video_id) + '/order_and_info.pkl', 'wb') as f:
                pickle.dump(batch, f)

    with open('../data/processed_data/extracted_faces/video' + str(video_id) + '/video' + str(video_id) + '.pkl', 'wb') as f:
        pickle.dump(video_id, f)


if __name__ == '__main__':
    video_id = sys.argv[1]
    SaveFaceBatchAndInfo(video_id)
    pass

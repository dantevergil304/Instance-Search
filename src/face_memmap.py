from process_face_data import BatchGenerator
from natsort import natsorted

import numpy as np
import pickle
import glob


def save_face_mmap(video_id):
    print('\nProcessing video %d' % video_id)
    batches = []
    dim0 = 0

    print('Calculating shape of the first dimension')
    for i, batch in enumerate(BatchGenerator(video_id)):
        if isinstance(batch, np.ndarray):
            print('[+] Loading batch %d' % i)
            batches.append(batch)
            dim0 += batch.shape[0]
        else:
            print('[+] Saving order_and_info.pkl')
            with open('../data/processed_data/memmap_faces/video' + str(video_id) + '/order_and_info.pkl', 'wb') as f:
                pickle.dump(batch, f)

    print('[+] Saving dim0.pkl')
    with open('../data/processed_data/memmap_faces/video' + str(video_id) + '/dim0.pkl', 'wb') as f:
        pickle.dump(dim0, f)

    print('[+] Saving merged.buffer')
    merged = np.memmap('../data/processed_data/memmap_faces/video' + str(video_id) +
                       '/merged.buffer', dtype=np.float32, mode='w+', shape=(dim0, 224, 224, 3))

    idx = 0
    for i, batch in enumerate(batches):
        print('[+] Assign batch %d to merged.buffer' % i)
        merged[idx:idx+len(batch), :, :, :] = batch
        idx += len(batch)

    del merged


if __name__ == '__main__':
    for i in range(244):
        save_face_mmap(i)

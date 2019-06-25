from util import mean_max_similarity_audio

import json
import glob
import os
import sys
import numpy as np


def audio_based_searching(query_embeddings, audio_id_list):
    '''
    Params:
    - query_embeddings: list of embedding extracted from query audios
    - audio_id_set: audio test list 
    '''

    result = []
    audio_feat_root_path = '../features/VGGish/embedding'
    for idx, audio_id in enumerate(audio_id_list):
        print(f'{idx} - processing audio {audio_id}')
        video_id = audio_id.split('_')[0][4:] 

        audio_embedding_path = os.path.join(audio_feat_root_path, f'video{video_id}', f'{audio_id}.npy')
        if os.path.exists(audio_embedding_path):
            audio_embedding = np.load(audio_embedding_path)
            result.append((audio_id, mean_max_similarity_audio(query_embeddings, audio_embedding)))
        else:
            result.append((audio_id, 0))

    result = sorted(result, key=lambda x: x[1], reverse=True)

    return result


if __name__ == '__main__':
    topics_data_folder = '../data/raw_data/queries/person-action-2019'
    topics_feat_folder = '../3rd_party/vggish/feature'
    config_folder = '../result/config_vggface2_2019_linear_svm_vgg16_pool5_gap'
    final_result_folder = f'{config_folder}_with_audio'

    query_ids = ['9261', '9262', '9274', '9266', '9276', '9265', '9275'] 
    query_ids = ['9262', '9274', '9266', '9276', '9265', '9275'] 

    for query_id in query_ids:
        # Get query embeddings
        action_examples_path  = os.path.join(topics_data_folder, query_id, 'video_action')

        query_embeddings = []
        for clip_path in glob.glob(os.path.join(action_examples_path, '*mp4')):
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]

            query_embeddings.append(np.load(os.path.join(topics_feat_folder, f'{clip_name}.npy')))


        person = os.path.basename(glob.glob(os.path.join(topics_data_folder, query_id, 'person_img', '*'))[0]).split('.')[0]

        person_result_path = os.path.join(config_folder, person, 'stage 1', 'result.txt')

        audio_id_list = []
        with open(person_result_path, 'r') as f:
            for line in f:
                _, _, audio_id, _, _, _ = line.rstrip().split()
                audio_id_list.append(audio_id)


        result = audio_based_searching(query_embeddings, audio_id_list)
            
        save_path = os.path.join(final_result_folder, query_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'result.txt'), 'w') as f:
            for i, r in enumerate(result):
                f.write(' '.join((query_id, 'Q0', r[0], str(i+1), str(r[1]), 'STANDARD')) + '\n')


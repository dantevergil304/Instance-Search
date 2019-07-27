from util import mean_max_similarity_action

import glob
import os
import numpy as np
import sys


def action_based_searching(query_embeddings, shot_id_list):
    '''
    Params:
    - query_embeddings: list of embedding extracted from query audios
    - shot_id_list: shot test list 
    '''
    result = []
    action_feat_root_path = '../features/C3D/fc6'

    for idx, shot_id in enumerate(shot_id_list):
        print(f'{idx} - processing audio {shot_id}')
        video_id = shot_id.split('_')[0][4:]

        action_embedding_path = os.path.join(
            action_feat_root_path, f'video{video_id}', f'{shot_id}.npy')

        if os.path.exists(action_embedding_path):
            action_embedding = np.load(action_embedding_path)
            l2_norm_action_embedding = action_embedding / \
                np.linalg.norm(action_embedding)
            result.append((shot_id, mean_max_similarity_action(
                query_embeddings, l2_norm_action_embedding)))
        else:
            result.append((shot_id, 0))

    result = sorted(result, key=lambda x: x[1], reverse=True)

    return result


if __name__ == '__main__':
    topics_data_folder = '../data/raw_data/queries/person-action-2019'
    topics_feat_folder = '../features/Query_feature/2019/c3d/fc6'
    config_folder = '../result/config_vggface2_2019_linear_svm_vgg16_pool5_gap_with_example_video'
    final_result_folder = f'{config_folder}_with_action'

    # query_ids = ['9249', '9250', '9251', '9252', '9253', '9254', '9255', '9256', '9257', '9258', '9259', '9260', '9261', '9262',
                 # '9263', '9264', '9265', '9266', '9267', '9268', '9269', '9270', '9271', '9272', '9273', '9274', '9275', '9276', '9277', '9278']
    query_ids = [sys.argv[1]]

    for query_id in query_ids:
        # Get query embeddings
        action_examples_path = os.path.join(
            topics_data_folder, query_id, 'video_action')

        query_embeddings = []
        for clip_path in glob.glob(os.path.join(action_examples_path, '*mp4')):
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]

            query_embedding = np.load(os.path.join(
                topics_feat_folder, f'{clip_name}.npy'))
            l2_norm_query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embeddings.append(l2_norm_query_embedding)

        person = os.path.basename(glob.glob(os.path.join(
            topics_data_folder, query_id, 'person_img', '*'))[0]).split('.')[0]

        person_result_path = os.path.join(
            config_folder, person, 'stage 1', 'result.txt')

        shot_id_list = []
        with open(person_result_path, 'r') as f:
            for line in f:
                _, _, audio_id, _, _, _ = line.rstrip().split()
                shot_id_list.append(audio_id)

        result = action_based_searching(query_embeddings, shot_id_list)

        save_path = os.path.join(final_result_folder, query_id, 'stage 1')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'result.txt'), 'w') as f:
            for i, r in enumerate(result):
                f.write(' '.join((query_id, 'Q0', r[0], str(
                    i+1), str(r[1]), 'STANDARD')) + '\n')

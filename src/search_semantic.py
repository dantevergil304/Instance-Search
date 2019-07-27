from util import mean_max_similarity_semantic
from natsort import natsorted

import numpy as np
import os
import glob
import sys


def L2_norm_semantic(semantic_embedding):
    ret = []
    for embedding in semantic_embedding:
        ret.append(embedding / np.linalg.norm(embedding))
    return np.array(ret)


def semantic_based_searching(topic_embeddings, shot_id_list):
    result = []
    semantic_feat_root_path = '../features/VGG19-1K'

    for idx, shot_id in enumerate(shot_id_list):
        print(f'{idx} - processing {shot_id}')
        video_id = shot_id.split('_')[0][4:]

        semantic_embedding = []
        for semantic_path in natsorted(glob.glob(os.path.join(semantic_feat_root_path, f'video{video_id}', f'{shot_id}', '*npy'))):
            semantic_embedding.append(np.load(semantic_path))

        l2_norm_semantic_embedding = L2_norm_semantic(semantic_embedding)
        result.append((shot_id, mean_max_similarity_semantic(
            topic_embeddings, l2_norm_semantic_embedding)))

    result = sorted(result, key=lambda x: x[1], reverse=True)

    return result


if __name__ == '__main__':
    topics_data_folder = '../data/raw_data/queries/person-action-2019'
    topics_feat_folder = '../features/Query_feature/2019/vgg19-1K'
    config_folder = '../result/config_vggface2_2019_linear_svm_vgg16_pool5_gap_with_example_video'
    final_result_folder = f'{config_folder}_with_semantic'

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
            l2_norm_query_embedding = L2_norm_semantic(query_embedding)
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

        result = semantic_based_searching(query_embeddings, shot_id_list)

        save_path = os.path.join(final_result_folder, query_id, 'stage 1')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'result.txt'), 'w') as f:
            for i, r in enumerate(result):
                f.write(' '.join((query_id, 'Q0', r[0], str(
                    i+1), str(r[1]), 'STANDARD')) + '\n')

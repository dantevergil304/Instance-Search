from scipy import stats
from natsort import natsorted

import os
import glob



def norm_zscore_sum_rule(result_list_a, result_list_b):
    '''
    result_list: list of tuple (shot_id, sim_score)
    '''
    result_list_a = sorted(result_list_a, key=lambda x: x[0])
    result_list_b = sorted(result_list_b, key=lambda x: x[0])

    shot_id_a, score_a = zip(*result_list_a)
    shot_id_b, score_b = zip(*result_list_b)

    final_score = 0.5*stats.zscore(score_a) + 0.5*stats.zscore(score_b)

    return sorted(list(zip(shot_id_a, final_score)), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    config_a = '../result/config_vggface2_2019_linear_svm_vgg16_pool5_gap_with_action/'
    config_b = '../result/config_vggface2_2019_linear_svm_vgg16_pool5_gap_with_semantic/'
    config_fusion = '../result/config_vggface2_2019_linear_svm_vgg16_pool5_gap_with_action_and_semantic/'

    for query_dir in natsorted(glob.glob(os.path.join(config_a, '*'))):

        query_name = os.path.basename(query_dir)
        print(f'processing {query_name}')

        result_list_a = []
        with open(os.path.join(config_a, query_name, 'stage 1', 'result.txt'), 'r') as f:
            for line in f:
                _, _, shot_id, _, score, _ = line.rstrip().split()
                result_list_a.append((shot_id, float(score)))

        result_list_b = []
        with open(os.path.join(config_b, query_name, 'stage 1', 'result.txt'), 'r') as f:
            for line in f:
                _, _, shot_id, _, score, _ = line.rstrip().split()
                result_list_b.append((shot_id, float(score)))


        result_list = norm_zscore_sum_rule(result_list_a, result_list_b)

        save_path = os.path.join(config_fusion, query_name, 'stage 1')
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, 'result.txt'), 'w') as f:
            for i, r in enumerate(result_list):
                f.write(' '.join((query_dir, 'Q0', r[0], str(
                 i+1), str(r[1]), 'STANDARD')) + '\n')

for i in $(ls ../../result/config_fc7_2018_peking)
do
    echo ${i}
    ../../../trec_eval/trec_eval -m map ../../../visualize-tools-master/person_gt_final.txt ../../result/config_fc7_2018_linear_svm_vgg16_pool5_gap/${i}/stage\ 2/result.txt
   
done

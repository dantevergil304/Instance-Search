for i in $(ls ../../result/config_fc7_2018_peking)
do
    echo ${i}
    ../../../trec_eval/trec_eval -m P.500 ../../../visualize-tools-master/person_gt_final.txt ../../result/config_fc7_2018_manual_rmBadFaces_2/${i}/stage\ 1/result.txt
   
done

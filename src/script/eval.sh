for i in $(ls ../../result/config_max_mean)
do
    echo ${i}
    echo "MAX_MEAN"
    ../../../trec_eval/trec_eval -m map ../../../Topics\ and\ GTs/gt_final.txt ../../result/config_max_mean/${i}/stage_1_trec_eval.txt
    echo "MEAN_MAX"
    ../../../trec_eval/trec_eval -m map ../../../Topics\ and\ GTs/gt_final.txt ../../result/config_1/${i}/stage_1_trec_eval.txt
    echo "MAX_MAX"
    ../../../trec_eval/trec_eval -m map ../../../Topics\ and\ GTs/gt_final.txt ../../result/config_max_max/${i}/stage_1_trec_eval.txt
   
    echo "\n" 
done

for i in $(ls ../../result/config_max_mean)
do
    echo ${i}
    ../../../trec_eval/trec_eval -m map ../../../Topics\ and\ GTs/gt_final.txt ../../result/config_1/${i}/stage_1_trec_eval.txt
    
done

export CUDA_VISIBLE_DEVICES=0,2

for video_id in $(ls -d ../../data/raw_data/shot-audios/*)
do
    echo $video_id
    for shot_wav in $(ls -d ${video_id}/*)
    do
        python vggish_inference_demo.py --wav_file ../../data/raw_data/shot-audios/video0/shot0_3.wav
    done
done

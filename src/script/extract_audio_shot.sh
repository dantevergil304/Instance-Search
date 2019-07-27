vid_id=${1}
for video_dir in $(ls -d ../../data/raw_data/shots/video${vid_id})
do
    video_id=$(basename ${video_dir})
    echo ${video_id}
    for shot in $(ls -d ${video_dir}/*)
    do
        shot_id=$(basename ${shot})
        shot_id="${shot_id%.*}"
        # echo ${shot_id}
        ffmpeg -i ${shot} ../../data/raw_data/shot-audios/${video_id}/${shot_id}.wav
    done
done

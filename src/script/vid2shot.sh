#!/bin/bash
video_list="../../data/raw_data/info/video_list.txt"
SECONDS=0
while IFS= read -r video_name 
do
    
    file="../../data/raw_data/info/master_shot_reference.txt"
    found=false
    echo "Searching $video_name in master_shot_reference.txt ..."

    while IFS= read -r line
    do
	IFS="    "
	set -- $line

	if [ $video_name != $1 ]
	then
	    continue
	fi

	if [ "$found" = false ]
	then
	    echo "File $video_name has been found."
	    found=true
	fi

	#name="../../data/raw_data/videos/$1"
	name="/home/hieudvm/videos/$1"
	start_time="$3"
	end_time="$4"
	output="../../data/raw_data/shots/$2.mp4"

	if [ ! -f $output ]
	then
	    ffmpeg -nostdin -ss $start_time -i $name -to $end_time -c:v libx264 -c:a aac -copyts $output
	fi

	# echo "!Warning: Path $output is already existed"
	# echo $name
    done <"$file"

done <"$video_list"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

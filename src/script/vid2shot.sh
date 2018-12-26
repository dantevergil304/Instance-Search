#!/bin/bash
file="../../data/raw_data/info/master_shot_reference.txt.sample.1"
while IFS= read -r line
do
	IFS="    "
	set -- $line
    name="../../data/raw_data/videos/$1"
    start_time="$3"
    end_time="$4"
    output="../../data/raw_data/shots/$2.mp4"
    echo $name
    ffmpeg -nostdin -ss $start_time -i $name -to $end_time -c:v libx264 -c:a aac -copyts $output
done <"$file"

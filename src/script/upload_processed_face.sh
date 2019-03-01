cd ..
file="../data/processed_data/extracted_faces/links.txt"
vid=158
while IFS= read -r line
#while : ; do
do
    echo "Processing Video ${vid}"
    python process_face_data.py ${vid}

    for batch in $(find ../data/processed_data/extracted_faces/video${vid} -type f -name '*')
    do
	echo "Batch $batch"
	while : ; do
	    output=$(gdrive upload -p ${line} $batch)
	    #output=$(gdrive upload -r -p 1j4erN74Q6lpb6QL5QQ39ZFSXb7zivhpA ../data/processed_data/extracted_faces/video${vid})
	    if [[ $output != *"Error 403"* ]]; then
		echo "\tUpload Successfully!\n"
		break
	    fi
	    echo "\tUpload Failed, Start reuploading\n"
	done

    done

    rm ../data/processed_data/extracted_faces/video${vid}/*
    vid=$((vid+1))
    
done <"$file"

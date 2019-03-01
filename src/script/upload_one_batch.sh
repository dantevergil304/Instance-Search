vid=$1
line=$2
echo "Video id $vid"
echo "Link id $line"
while : ; do
    output=$(gdrive upload -p ${line} ../data/processed_data/extracted_faces/video${vid}/*)
    #output=$(gdrive upload -r -p 1j4erN74Q6lpb6QL5QQ39ZFSXb7zivhpA ../data/processed_data/extracted_faces/video${vid})
    if [[ $output != *"Error 403"* ]]; then
	echo "\tUpload Successfully!\n"
	break
    fi
    echo "\tUpload Failed, Start reuploading\n"
done
rm ../data/processed_data/extracted_faces/video${vid}/*

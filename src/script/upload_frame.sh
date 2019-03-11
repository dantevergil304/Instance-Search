for i in {0..244}
do
	echo "[+] Processing video$i\n"
	mkdir ~/video$i

	echo "\tStart getting all frames of video$i...\n"
	find . -name "~/Instance_Search/data/processed_data/frames/shot$i_*" -exec cp -r {} ~/video$i \;

	echo "\tStart Compressing video$i directory...\n"
	zip -r ~/video$i.zip ~/video$i
	echo "\tFinished compressing\n"

	while : ; do
		output=$(gdrive upload -p 1MCaHhP4M3jw_9b5ARLjjmTeQrCkvlvKz ~/video$i.zip)
		if [[ $output != *"Error 403"* ]]; then
			echo "\tUpload Successfully!\n"
			break	
		fi
		echo "\tUpload Failed, Start reuploading\n"
	done

	rm -rf ~/video$i
	rm ~/video$i.zip
done

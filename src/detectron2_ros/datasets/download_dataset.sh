#! /bin/bash


#here should be the links of dataset
links=(
"https://sandbox.zenodo.org/record/1181873/files/lvis.zip?download=1"
)

for i in "${links[@]}"
do
	#download the file if it's not in the folder and there is no folder with the same name
	file_name=$(basename "$i")
	folder_name="${file_name%.*}"
	echo "$file_name"
	echo "$folder_name" #datasets
	#if file and folder don't exist,download the dataset(zip file)
	if [ ! -f "$file_name" ] && [ ! -d "$folder_name" ]; then
		echo "step1"
		wget "$i"
	fi
	#if zip file already existed,and the folder doesn't exist
	if [ -f "$file_name" ] && [ ! -d "$folder_name" ]; then
		echo "step2"
		#extract the file
		unzip "$file_name"
		#remove the zip file
		rm "$file_name"
	fi
done

	

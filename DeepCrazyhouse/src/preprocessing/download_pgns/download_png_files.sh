#!/bin/bash
# Downloads all pgn files which are used for supervised training (train, validation, test)
# See configs/main_config_sample.py which dataset-type used wich pgns
file="./pgn_list_lichess_variants.txt"
while IFS= read -r line
do
filename=$(echo $line| cut -d'/' -f 5)
filepgn=${filename::${#filename}-4}

if [[ -f "${PWD}/${filepgn}" ]]; then
	echo "Already downloaded $filepgn"
else
	printf 'Downloading file %s ...\n' "$line"
if curl --silent -o "${PWD}/${filename}" -L "$line"; then
     bunzip2 "${filename}"
     if [[ -f "${PWD}/${filename}" ]]; then
         echo "Removing file ${filename}"
         rm -f "${PWD}/${filename}"
     fi
else
     echo "Something went wrong"
fi
fi
done <"$file"
printf 'Download finished sucessfully!\n'

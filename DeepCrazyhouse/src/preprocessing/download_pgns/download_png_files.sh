#!/bin/bash
# Downloads all pgn files which are used for supervised training (train, validation, test)
# See configs/main_config_sample.py which dataset-type used wich pgns
file="./pgn_list_lichess_variants.txt"
while IFS= read -r line
do
	printf 'Downloading file %s ...\n' "$line"
	curl -O "$line"
done <"$file"
printf 'Download finished sucessfully!\n'

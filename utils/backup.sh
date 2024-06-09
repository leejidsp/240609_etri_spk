#!/bin/bash

# Command: ./backup.sh exp_name mode
exp=$1	#exp_name
mode=$2	#train/test
FILE="./log/${exp}.log"

# Backup commands
if [ -z "${exp}" ]; then
	echo "ERROR: Argument(exp name) is not given."
	exit 1
elif [ ! -f "${FILE}" ]; then
	echo ${FILE}
	echo "ERROR: Incorrect exp name."
	exit 1
else
	# Train
	if [ "${mode}" == "train" ]; then 
		mkdir backup/${exp}
		mkdir backup/${exp}/configs
		cp *.py ./backup/${exp}/
		cp -r models ./backup/${exp}/
		cp -r trainers ./backup/${exp}/
		cp -r losses ./backup/${exp}/
		cp -r data ./backup/${exp}/
		cp -r utils ./backup/${exp}/
		cp configs/${exp}.yaml ./backup/${exp}/configs/
		cp -r configs/data ./backup/${exp}/configs/
		cp -r configs/loss ./backup/${exp}/configs/
		cp -r configs/model ./backup/${exp}/configs/
		cp -r configs/opt ./backup/${exp}/configs/
		echo "Backup finished."
		exit 0
	# Test
	elif [ "${mode}" == "train_finished" ]; then 
		cp log/${exp}.log ./backup/${exp}/
		cp log/${exp}.summary ./backup/${exp}/
		echo "Backup finished."
	else
		echo "ERROR: Incorrect mode."
		exit 1
	fi
fi


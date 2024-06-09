#!/bin/bash

exp=$1
FILE="./log_txt/${exp}.log"

if [ -z "${exp}" ]; then
	echo "ERROR: Argument(exp name) is not given."
	exit 1
elif [ ! -f "${FILE}" ]; then
	echo "Log file (${FILE}) doesn't exist."
	echo "ERROR: Incorrect exp name."
	exit 1
else	
	rm -rf checkpoint/${exp}
	rm -rf tb/${exp}
	rm -rf log/${exp}.log
	rm -rf log/${exp}.summary
	rm -rf backup/${exp}
	rm -rf eval_samples/LibriSpeech_tb_samples/${exp}
	echo "EXP ${exp} is removed."
	exit 0
fi

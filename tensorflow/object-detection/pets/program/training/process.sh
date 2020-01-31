#!/bin/bash

CONFIG_FILE="/tmp/config_file/pipeline.config"
mkdir -p "/tmp/config_file/"
DEFAULT_FILEPATH="./pipeline.config"

if [ ! -z "$DKUBE_JOB_CONFIG_FILE" ]; then
    echo "Using config file from parameters"
    cp "$DKUBE_JOB_CONFIG_FILE" "$CONFIG_FILE"
elif [ -f "$DEFAULT_FILEPATH" ]; then
    echo "using config file from workspace"
    cp "$DEFAULT_FILEPATH" "$CONFIG_FILE"
else
    echo "No config file provided"
    exit 1
fi

echo "Config file path : $CONFIG_FILE"

DATA_DIR="/opt/dkube/input/dataset"

MODEL_DIR="/opt/dkube/input/model"

#Set datset path in pipeline config file
sed -i "s|DATA_PATH|"${DATA_DIR}"|g" $CONFIG_FILE

EXTRACT_PATH="/tmp/object-detection"
mkdir -p $EXTRACT_PATH
echo "Extract path : $EXTRACT_PATH"

#Extract model
for file in $MODEL_DIR/*; do
	filename=$(basename -- "$file")
	extension="${filename##*.}"
        echo "Model file format : $extension"
	if [[ $extension == "gz" ]]; then
        	echo "Extracting model"
                tar -xvf $file -C $EXTRACT_PATH
	elif [[ $extension == "zip" ]]; then
        	echo "Extracting model"
                unzip $file -d $EXTRACT_PATH
	else
        	echo "Unsupported format"
		exit 1
	fi
done

#Set the model path in pipeline.config file to the extracted path
sed -i "s|MODEL_PATH|"${EXTRACT_PATH}"|g" $CONFIG_FILE

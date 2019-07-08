#!/bin/bash

CONFIG_FILE="/tmp/config_file/pipeline.config"
mkdir -p "/tmp/config_file/"
DEFAULT_FILEPATH="./pipeline.config"

if [ -n "$CONFIG_FILEPATH" ]; then
    echo "Using config file from parameters"
    cp "$CONFIG_FILEPATH" "$CONFIG_FILE"
elif [ -f "$DEFAULT_FILEPATH" ]; then
    echo "using config frile from workspace"
    cp "$DEFAULT_FILEPATH" "$CONFIG_FILE"
else
    echo "No config file provided"
    exit 1
fi

echo "Config file path : $CONFIG_FILE"

DATA_DIR="${DATUMS_PATH}/${DATASET_NAME}"
MODEL_DIR="${MODEL_PATH}/${MODEL_NAME}"

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
sed -i '/num_steps/c\  num_steps : '"${TF_TRAIN_STEPS}"'' $CONFIG_FILE

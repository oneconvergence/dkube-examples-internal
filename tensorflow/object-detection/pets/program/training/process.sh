#!/bin/bash

CONFIG_FILE="/tmp/config_file/pipeline.config"
mkdir -p "/tmp/config_file/"
DEFAULT_FILEPATH="./pipeline.config"

if [ ! -z "$HYPERPARAMS_FILEPATH" ]; then
    echo "Using config file from parameters"
    cp "$HYPERPARAMS_FILEPATH" "$CONFIG_FILE"
elif [ -f "$DEFAULT_FILEPATH" ]; then
    echo "using config file from workspace"
    cp "$DEFAULT_FILEPATH" "$CONFIG_FILE"
else
    echo "No config file provided"
    exit 1
fi

echo "Confg file path : $1"

DATA_DIR="${DATUMS_PATH}/${DATASET_NAME}"
MODEL_DIR="${MODEL_PATH}/${MODEL_NAME}"

#Set datset path in pipeline config file
sed -i "s|DATA_PATH|"${DATA_DIR}"|g" $1

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
sed -i "s|MODEL_PATH|"${EXTRACT_PATH}"|g" $1
sed -i '/num_steps/c\  num_steps : '"${TF_TRAIN_STEPS}"'' $1

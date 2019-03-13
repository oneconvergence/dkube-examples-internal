#!/bin/bash
DATA_DIR="${DATUMS_PATH}/${DATASET_NAME}"
MODEL_DIR="${MODEL_PATH}/${MODEL_NAME}"
cp ./pipeline.config $HOME/pipeline.config
echo "Confg file path : $HOME/pipeline.config"

if [ $# -ne 1 ]; then
    echo "Usage : bash process.sh <c/n>"
    exit 1
elif [ $1 = "n" ]; then
	sed -i "s|DATA_PATH|"${DATA_DIR}"|g" $HOME/pipeline.config
	sed -i "s|MODEL_PATH|"${MODEL_DIR}"|g" $HOME/pipeline.config
elif [ $1 = "c" ]; then
	EXTRACT_PATH="/tmp/object-detection"
	mkdir -p $EXTRACT_PATH
	echo "Extract path : $EXTRACT_PATH"

	#Extract datasets
	for file in $DATA_DIR/*; do
		filename=$(basename -- "$file")
		extension="${filename##*.}" 
		echo "Dataset file format : $extension"
		if [[ $extension == "gz" ]]; then
			echo "Extacting dataset"
			tar -xvf $file -C $EXTRACT_PATH
		elif [[ $extension == "zip" ]]; then
			echo "Extracting dataset"
			unzip $file -d $EXTRACT_PATH
		else
			echo "Unsupported format"
			exit 1
		fi
	done

	#Extract models
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

	sed -i "s|DATA_PATH|"${EXTRACT_PATH}"|g" $HOME/pipeline.config
	sed -i "s|MODEL_PATH|"${EXTRACT_PATH}"|g" $HOME/pipeline.config
fi

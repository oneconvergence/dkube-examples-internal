#!/bin/bash
PVC_HOST_MOUNT_PATH="/book-classification"
DATA_URL="https://github.com/oneconvergence/dkube-examples/blob/book-classification/kubeflow/book-classification/data/data_pipeline_data.csv?raw=true"
UTILS_SCRIPT_URL="https://github.com/oneconvergence/dkube-examples/blob/book-classification/kubeflow/book-classification/tfx/utils.py?raw=true"

create_mount_path()
{
	if [ ! -d "$PVC_HOST_MOUNT_PATH" ]
	then
		echo "Mount path doesn't exist. Creating it with required data."
	    mkdir -p $PVC_HOST_MOUNT_PATH/data
	    mkdir -p $PVC_HOST_MOUNT_PATH/tfx
	    wget $DATA_URL -O $PVC_HOST_MOUNT_PATH/data/data.csv
		wget $UTILS_SCRIPT_URL -O $PVC_HOST_MOUNT_PATH/tfx/utils.py
	    echo "Created mount path along with required data"
	else
	    echo "Mount path already exists"
	fi
}

run_clean_up()
{
	out=`rm -rf $PVC_HOST_MOUNT_PATH/output`
	out=`kubectl -n kubeflow exec -it $(kubectl -n kubeflow get pods -l app=mysql --no-headers -o custom-columns=:metadata.name) -- mysql --user root --execute 'drop database mlmd_book_classification_db'`
}

create_volume()
{
	echo 'apiVersion: v1
kind: PersistentVolume
metadata:
  name: book-classification
  labels:
    type: local
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: $PVC_HOST_MOUNT_PATH' >> /tmp/pv.yaml

	echo 'apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: book-classification-claim
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi' >> /tmp/pvc.yaml

	out=`kubectl apply -f /tmp/pv.yaml`
	out=`kubectl apply -f /tmp/pvc.yaml -n kubeflow`
	echo "created pv with name book-classification"
	echo "created pvc with name book-classification-claim"
}

get_mysql_ip(){
	out=`kubectl -n kubeflow get pod $(kubectl -n kubeflow get pods -l app=mysql --no-headers -o custom-columns=:metadata.name) --template={{.status.podIP}}`
	echo "mysql is running at ip address $out, please use this ip address in KUBEFLOW_MD_DB_SERVER variable while running pipeline from notebook"
}

setup() {
	run_clean_up
	create_mount_path
	create_volume
	get_mysql_ip
}

cleanup() {
	run_clean_up
}

if [[ $1 == "setup" ]]
then
	setup
elif [[ $1 == "cleanup" ]]
then
	cleanup
else
	echo "please provide either 'setup' or 'cleanup' as argument"
fi

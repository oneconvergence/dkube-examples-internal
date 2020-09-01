############### Importing Libraries ##############
import pandas as pd
import requests
import pickle
import json
import os
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , log_loss

############# Mount Paths ############
if os.getenv('DKUBE_JOB_CLASS',None) == 'notebook':
    MODEL_DIR = "model"
    DATA_DIR = "/opt/dkube/input"
    if not os.path.exists('model'):
        os.makedirs('model')
else:
    MODEL_DIR = "/opt/dkube/output"
    DATA_DIR = "/opt/dkube/input"

def read_data(csv_file):
	data=pd.read_csv(csv_file)
	return data

def get_metrics(actual, pred):
	accuracy=accuracy_score(actual,pred)
	loss=log_loss(actual,pred)
	return accuracy,loss

def api_calling(train_loss,train_accuracy,eval_loss,eval_accuracy):
	url="http://dkube-exporter.dkube:9401/export-training-info"
	train_metrics={}
	eval_metrics={}
	############# Train metrics ###############
	train_metrics['mode']="train"
	train_metrics['loss']=train_loss
	train_metrics['accuracy']=train_accuracy
	train_metrics['epoch']=1
	train_metrics['step']=1
	train_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
	train_metrics['jobuuid']=os.getenv('DKUBE_JOB_UUID')
	train_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
	train_metrics['max_steps']="1"
	requests.post(url, data=json.dumps({'data': [train_metrics]}))
	############ Evaluation metrics ###########
	eval_metrics['mode']="eval"
	eval_metrics['loss']=eval_loss
	eval_metrics['accuracy']=eval_accuracy
	eval_metrics['epoch']=1
	eval_metrics['step']=1
	eval_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
	eval_metrics['jobuuid']=os.getenv('DKUBE_JOB_UUID')
	eval_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
	eval_metrics['max_steps']="1"
	requests.post(url, data=json.dumps({'data': [eval_metrics]}))	

if __name__ == "__main__":
	######## Training ###########
	train_data=read_data(DATA_DIR +'/train_data_heart.csv')
	x_train=train_data.iloc[:,:-1].values
	y_train=train_data.iloc[:,13].values
	rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
	rf.fit(x_train, y_train)
	######## Model Saving ########
	filename = MODEL_DIR + '/model.joblib'
	joblib.dump(rf, filename)
	######## Training Metrics #######
	y_pred_train=rf.predict(x_train)
	train_accuracy,train_loss=get_metrics(y_train,y_pred_train)
	######## Evaluation Metrics ########
	test_data=read_data(DATA_DIR+'/test_data_heart.csv')
	x_test=test_data.iloc[:,:-1].values
	y_test=test_data.iloc[:,13].values
	y_pred=rf.predict(x_test)
	eval_accuracy,eval_loss=get_metrics(y_test,y_pred)
	####### API Calling ########
	api_calling(train_loss,train_accuracy,eval_loss,eval_accuracy)

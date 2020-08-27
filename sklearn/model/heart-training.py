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
DATA_DIR = '/opt/dkube/input'
MODEL_DIR = '/opt/dkube/model'

def read_data(csv_file):
	data=pd.read_csv(csv_file)
	return data

def eval_metrics(actual, pred):
	accuracy=accuracy_score(actual,pred)
	loss=log_loss(actual,pred)
	return accuracy,loss

if __name__ == "__main__":
	metrics={}
	url="http://dkube-exporter.dkube:9401/export-training-info"
	######## Training ###########
	train_data=read_data(DATA_DIR +'/train_data_heart.csv')
	x_train=train_data.iloc[:,:-1].values
	y_train=train_data.iloc[:,13].values
	rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
	rf.fit(x_train, y_train)
	######## Model Saving ########
	filename = MODEL_DIR + '/model.joblib'
	joblib.dump(rf, filename)
	######## Evaluation ########
	test_data=read_data(DATA_DIR+'/test_data_heart.csv')
	x_test=test_data.iloc[:,:-1].values
	y_test=test_data.iloc[:,13].values
	y_pred=rf.predict(x_test)
	accuracy,loss=eval_metrics(y_test,y_pred)
	####### API Calling ########
	metrics['mode']="eval"
	metrics['loss']=loss
	metrics['accuracy']=accuracy
	metrics['epoch']=1
	metrics['step']=1
	metrics['jobid']=os.getenv('DKUBE_JOB_ID')
	metrics['jobuuid']=os.getenv('DKUBE_JOB_UUID')
	metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
	metrics['max_steps']="1"
	#metrics['max_epochs']="1"
	requests.post(url, data=json.dumps({'data': [metrics]}))






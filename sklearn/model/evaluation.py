import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv, sys
import joblib
import requests, os

def log_metrics(key, value):
    url = "http://dkube-exporter.dkube:9401/mlflow-exporter"
    train_metrics = {}
    train_metrics['mode']="train"
    train_metrics['key'] = key
    train_metrics['value'] = value
    train_metrics['epoch'] = 1
    train_metrics['step'] = 1
    train_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
    train_metrics['run_id']=os.getenv('DKUBE_JOB_UUID')
    train_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
    requests.post(url, json = train_metrics)

dates = []
prices = []


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

DATA_DIR = '/opt/dkube/input/'
MODEL_DIR = '/opt/dkube/model/'


def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return



if __name__ == "__main__":

    get_data(DATA_DIR +'goog.csv')

    dates = np.reshape(dates,(len(dates), 1))

    svm = joblib.load(MODEL_DIR + 'model.joblib') 

    predictions = svm.predict(dates)

    (rmse, mae, r2) = eval_metrics(prices, predictions)
    log_metrics('RMSE', rmse)
    log_metrics('MAE', mae)
    log_metrics('R2', r2)
    print('RMSE', rmse)
    print('R2', r2)
    print('MAE', mae)


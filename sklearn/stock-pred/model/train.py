import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv, sys
import matplotlib.pyplot as plt
from tensorboard_logger import unconfigure, configure, log_value, log_histogram, log_images,  Logger
from PIL import Image
import cv2, os, json
import joblib
import requests
import argparse

dates = []
prices = []
unconfigure()

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

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

DATA_DIR = '/opt/dkube/input'
MODEL_DIR = '/opt/dkube/model'

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

if not os.path.exists(MODEL_DIR + "/logs/SVMrun"):
    os.makedirs(MODEL_DIR + "/logs/SVMrun")

configure(MODEL_DIR + "/logs/SVMrun", flush_secs=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--name',type=str,help='name',default='SVM for stock Prediction')
    parser.add_argument('--kernel',type=str,help='kernel type',default='rbf')
    parser.add_argument('--C',type=float,help='penality parameter for the error term',default=1e3)
    parser.add_argument('--gamma',type=float,help='gamma parameter',default=0.1)
    parser.add_argument('--degree',type=int,help='degree of polynomial kernel function',default=2)
    global FLAGS
    FLAGS,unparsed=parser.parse_known_args()
    name = FLAGS.name
    kernel = FLAGS.kernel
    C=FLAGS.C
    gamma = FLAGS.gamma
    degree= FLAGS.degree


    print ("MODEL_DIR:{}, DATA_DIR:{}".format(MODEL_DIR,DATA_DIR))
    get_data(DATA_DIR +'/goog.csv')
    dates = np.reshape(dates,(len(dates), 1))

    svm = SVR(kernel= kernel, C= C, degree= degree, gamma=gamma)
    svm.fit(dates, prices)

    predictions = svm.predict(dates)
    (rmse, mae, r2) = eval_metrics(prices, predictions)
    
    log_metrics('RMSE', rmse)
    log_metrics('MAE', mae)
    log_metrics('R2', r2)

    plt.plot(dates, prices, color= 'black', label = "Data", marker = '*')
    plt.plot(dates,predictions, color= 'red', label = "Predictions", marker = 'o')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SVM predictions with '+kernel+ ' kernel')
    plt.legend()
    plt.savefig('svm.png')

    log_value('RMSE', rmse)
    log_value('MAE', mae)
    log_value('R2', r2)
    filename = MODEL_DIR + '/model.joblib'	
    joblib.dump(svm, filename)

    img = cv2.imread('svm.png')
    log_histogram('Stock Prices', prices, step=1)
    log_images('Stock Predictions Graph',[img])


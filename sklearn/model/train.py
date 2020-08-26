import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv, sys
import matplotlib.pyplot as plt
from tensorboard_logger import unconfigure, configure, log_value, log_histogram, log_images,  Logger
from PIL import Image
import cv2, os, json
import joblib

dates = []
prices = []

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

DATA_DIR = '/opt/dkube/input'
MODEL_DIR = '/opt/dkube/model'
metric_path = MODEL_DIR + '/metrics/'

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

unconfigure()
configure(MODEL_DIR + "/logs/SVMrun", flush_secs=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SKLearn stock predicton example')
    parser.add_argument('--name', type=str, default="SVM for stock Preiction")
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--C', type=float, default=1e3)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--degree', type=float, default=2)
    args, _ = parser.parse_known_args()

    name = args.name
    kernel = args.kernel
    C = args.C
    gamma = args.gamma
    degree = args.degree


    print ("MODEL_DIR:{}, DATA_DIR:{}".format(MODEL_DIR,DATA_DIR))
    get_data(DATA_DIR +'/goog.csv')
    dates = np.reshape(dates,(len(dates), 1))

    svm = SVR(kernel= kernel, C= C, degree= degree, gamma=gamma)
    svm.fit(dates, prices)

    predictions = svm.predict(dates)
    (rmse, mae, r2) = eval_metrics(prices, predictions)
    
    metrics = []
    metric_names = ['rmse', 'mae', 'r2']
    train_metrics = [rmse, mae, r2]
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    for i in range(3):
        temp = {}
        temp['class'] = 'scalar'
        temp['name'] = metric_names[i]
        temp['value'] = str(train_metrics[i])
        metrics.append(temp)
    metrics = {'metrics':metrics}
    with open(metric_path + 'metrics.json', 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
            
    filename = MODEL_DIR + '/model.joblib'
    joblib.dump(svm, filename)

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

    img = cv2.imread('svm.png')
    log_histogram('Stock Prices', prices, step=1)
    log_images('Stock Predictions Graph',[img])

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv, sys
import joblib

dates = []
prices = []
name = str(sys.argv[1]) if len(sys.argv) > 1 else 'SVM for stock Preiction'
kernel = str(sys.argv[2]) if len(sys.argv) > 2 else 'rbf'
C = float(sys.argv[3]) if len(sys.argv) > 3 else 1e3
gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
degree= int(sys.argv[5]) if len(sys.argv) > 5 else 2

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
    print('RMSE', rmse)
    print('R2', r2)
    print('MAE', mae)
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import requests, os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import mlflow
from sklearn import metrics
from dkube.sdk import *

inp_path = "/opt/dkube/input"
out_path = "/opt/dkube/output"

if __name__ == "__main__":

    ########--- Read features from input FeatureSet ---########

    # Featureset API
    featureset = DkubeFeatureSet()
    # Specify featureset path
    featureset.update_features_path(path=inp_path)

    # Read features
    data = featureset.read()  # output: response json with data
    feature_df = data["data"]

    ########--- Train ---########
    insurance_input = feature_df.drop(['charges'],axis=1)
    insurance_target = feature_df['charges']
    
    #stadardize data
    x_scaled = StandardScaler().fit_transform(insurance_input)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                                    insurance_target,
                                                    test_size = 0.25,
                                                    random_state=1211)
    #fit linear model to the train set data
    linReg = LinearRegression()
    linReg_model = linReg.fit(x_train, y_train)
    
    y_pred_train = linReg.predict(x_train)    # Predict on train data.
    y_pred_train[y_pred_train < 0] = y_pred_train.mean()
    y_pred = linReg.predict(x_test)   # Predict on test data.
    y_pred[y_pred < 0] = y_pred.mean()
    
    #######--- Calculating metrics ---############
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    intersept = linReg.intercept_
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    print('Root Mean Squared Error:', rmse)
    print('Intercept: ', intersept)

    ########--- Logging metrics into Dkube via mlflow ---############
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("intercept", intersept)
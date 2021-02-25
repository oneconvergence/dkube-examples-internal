library(caret)
library(mlbench)
library(randomForest)
library(doMC)
library(httr)
library(mlflow)

IN_DIR <- "/opt/dkube/model/"

registerDoMC(cores=8)
# load dataset
data(Sonar)
set.seed(7)
# create 80%/20% for training and validation datasets
validation_index <- createDataPartition(Sonar$Class, p=0.80, list=FALSE)
validation <- Sonar[-validation_index,]
model <- readRDS(sprintf("%smodel.rds", IN_DIR))
print(model)
# make a predictions on "new data" using the final model
final_predictions <- predict(model, validation[,1:60])
result=confusionMatrix(final_predictions, validation$Class)
##### Retrieving the metrics ######
Prevalence <- result$byClass['Prevalence']    
Sensitivity <- result$byClass['Sensitivity']
accuracy <- result$byClass['Balanced Accuracy']
# Logging metrics
mlflow_log_metric("Prevalence", Prevalence)
mlflow_log_metric("Sensitivity", Sensitivity)
mlflow_log_metric("Accuracy", accuracy)

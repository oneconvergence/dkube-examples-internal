# load libraries
library(caret)
library(mlbench)
library(randomForest)
library(doMC)
library(mlflow)

OUT_DIR <- "/opt/dkube/model/"

registerDoMC(cores=8)
# load dataset
data(Sonar)
set.seed(7)
# create 80%/20% for training and validation datasets
validation_index <- createDataPartition(Sonar$Class, p=0.80, list=FALSE)
validation <- Sonar[-validation_index,]
training <- Sonar[validation_index,]
# create final standalone model using all training data
set.seed(7)
model <- randomForest(Class~., training, mtry=2, ntree=2000)
# Metrics
final_predictions <- predict(model, training[,1:60])
result=confusionMatrix(final_predictions, training$Class)
##### Retrieving the metrics ######
Prevalence <- result$byClass['Prevalence']    
Sensitivity <- result$byClass['Sensitivity']
accuracy <- result$byClass['Balanced Accuracy']
# Logging metrics
mlflow_log_metric("Prevalence", Prevalence)
mlflow_log_metric("Sensitivity", Sensitivity)
mlflow_log_metric("Accuracy", accuracy)
# save the model to disk
saveRDS(model, sprintf("%smodel.rds", OUT_DIR))

install.packages('hash')  
library(caret)
library(mlbench)
library(randomForest)
library(doMC)
library(httr)
library(hash)

IN_DIR <- "/opt/dkube/model/"

registerDoMC(cores=8)
# load dataset
data(Sonar)
set.seed(7)
# create 80%/20% for training and validation datasets
validation_index <- createDataPartition(Sonar$Class, p=0.80, list=FALSE)
validation <- Sonar[-validation_index,]
model <- readRDS(paste(IN_DIR, "model.rds"))
print(model)
# make a predictions on "new data" using the final model
final_predictions <- predict(model, validation[,1:60])
result=confusionMatrix(final_predictions, validation$Class)
##### Retrieving the metrics ######
precision <- result$byClass['Pos Pred Value']    
recall <- result$byClass['Sensitivity']
accuracy <- result$byClass['Balanced Accuracy']

#### Metric function #####
log_metrics<-function(key, value){
url <-"http://dkube-exporter.dkube:9401/mlflow-exporter"

train_metrics<-hash()
train_metrics[['mode']]<-"train"
train_metrics[['key']]<-key
train_metrics[['value']]<-value
train_metrics[['epoch']]<-1
train_metrics[['step']]<-1
train_metrics[['jobid']]<-Sys.getenv('DKUBE_JOB_ID')
train_metrics[['run_id']]<-Sys.getenv('DKUBE_JOB_UUID')
train_metrics[['username']]<-Sys.getenv('DKUBE_USER_LOGIN_NAME')
POST(url,body = train_metrics,encode="json")}

##### Logging the metrics #######
log_metrics('Precision',precision)
log_metrics('Recall',recall)
log_metrics('Accuracy',accuracy)


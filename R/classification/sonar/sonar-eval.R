library(caret)
library(mlbench)
library(randomForest)
library(doMC)

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
confusionMatrix(final_predictions, validation$Class)
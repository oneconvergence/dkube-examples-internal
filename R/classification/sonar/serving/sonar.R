library(methods)
library(randomForest)
library(caret)
library(mlbench)
library(doMC)

new_rf <- function(filename) {
  model <- readRDS(filename)
  structure(list(model=model), class = "sonar")

}

predict.sonar <- function(sonar,newdata=list()) {
  rf_model =sonar$model 
  res <- stats::predict(rf_model,newdata[["data"]])
  data.frame(res)
}
send_feedback.sonar <- function(sonar,request=list(),reward=1,truth=list()) {
}


#dkube-kfserving - take the filepath as input
initialise_seldon <- function(filename, params) {
  new_rf(filename)
}
                  
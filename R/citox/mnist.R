library(methods)
library(randomForest)
source('transformer.R')

predict.mnist <- function(mnist,newdata=list()) {
  rf_model =mnist$model 
  i <- c(5, 6)
  newdata[ , i] <- apply(newdata[ , i], 2,function(x) as.numeric(as.character(x)))
  feature_tab  <- newdata
  model_features <- t(apply(feature_tab[,1:2], 1, split2di))
  model_features <- as.data.frame(model_features)
  model_features$dg = unlist(feature_tab[,5])
  res <- stats::predict(rf_model,model_features)
  data.frame(res)
}
send_feedback.mnist <- function(mnist,request=list(),reward=1,truth=list()) {
}


#dkube-kfserving - take the filepath as input
initialise_seldon <- function(filename, params) {
  new_rf(filename)
}
                  

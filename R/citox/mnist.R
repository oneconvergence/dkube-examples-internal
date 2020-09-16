library(methods)
library(randomForest)


dinucs <- function(nucs = c('A','T','C','G'), sugars = c('L','D')){
  
  states.di <- apply(expand.grid(nucs, nucs,'_',sugars,sugars), 1, 
                     function(x) paste(x,collapse = ''))
  sort(states.di)
  
}

split2di <- function(x) {  
  
  alldi <- dinucs()
  
  a <- paste(substring(x[1],1:(nchar(x[1])-1),2:(nchar(x[1]))),
             substring(x[2],1:(nchar(x[2])-1),2:(nchar(x[2]))),sep='_')
  a <- factor(a,levels = alldi)
  table(a)
}


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

new_rf <- function(filename) {
  model <- readRDS(filename)
  structure(list(model=model), class = "mnist")

}

initialise_seldon <- function(filename, params) {
  new_rf(filename)
}
                  

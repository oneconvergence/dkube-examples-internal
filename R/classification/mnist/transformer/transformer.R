library(methods)
library(randomForest)
library(plumber)
library(jsonlite)
library(optparse)
library(urltools)
library(stringi)



preprocess <- function(json_data) {
result <- fromJSON(json_data)
df<-data.frame(result$inputs)
return df 
}

postprocess <- function(df){
df
}



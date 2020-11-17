library(methods)
library(randomForest)
library(plumber)
library(jsonlite)
library(optparse)
library(urltools)
library(stringi)

library(hash)
library(float)
library(utf8)
library(OpenImage)


preprocess <- function(json_data) {
  json_data$instances<-NULL
  data=json_data$signatures$inputs[[1]][[3]]
  image <- readImage("image.png")
  image <- resizeImage(image, width = 28, height = 28, method = 'nearest')
  image <- array(image[1:28,1:28], dim = c(1,784))
  instances<-list(image)
  token <- json_data["token"]
  payload<-hash()
  payload[["instances"]]=instances
  payload[["token"]]=token
  payload
}


postprocess <- function(df){
df
}



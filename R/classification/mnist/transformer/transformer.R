library(methods)
library(randomForest)
library(plumber)
library(jsonlite)
library(optparse)
library(urltools)
library(stringi)

library(hash)
library(OpenImageR)
library(reticulate)


preprocess <- function(json_data) {
  json_data$instances<-NULL
  data=json_data$signatures$inputs[[1]][[3]]
  source_python("get-image.py")
  get_image(data)
  image <- readImage("image.png")
  image <- array(image, dim = c(1,784))
  image <- image/255
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



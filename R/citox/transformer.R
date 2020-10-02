library(methods)
library(randomForest)
library(plumber)
library(jsonlite)
library(optparse)
library(urltools)
library(stringi)

extract_data <- function(jdf) {
  if ("ndarray" %in% names(jdf$data)){
    jdf$data$ndarray
  } else {
    data <- jdf$data$tensor$values
    dim(data) <- jdf$data$tensor$shape
    data
  }
}

extract_names <- function(jdf) {
  if ("names" %in% names(jdf$data)) {
    jdf$data$names
  } else {
    list()
  }
}

preprocess <- function(jdf) {
  data = extract_data(jdf)
  names = extract_names(jdf)
  df <- data.frame(data)
  colnames(df) <- names
  df
}

postprocess <- function(df){
  df
}
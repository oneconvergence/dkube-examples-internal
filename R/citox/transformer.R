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
                     
new_rf <- function(filename) {
  model <- readRDS(filename)
  structure(list(model=model), class = "mnist")

}
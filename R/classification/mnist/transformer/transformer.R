library(png)
library(base64enc)
library(readr)
function(base64string){
    outconn <- file("inp.png","wb")
    base64decode(what=base64string, output=outconn)
    close(outconn)
    x <- readPNG("inp.png")
    y <- array(x, c(1,784))
    y
}
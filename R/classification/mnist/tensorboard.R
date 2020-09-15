library(reticulate)
library(tensorflow)
# check
tl = import("tensorboard_logger")
np = import("numpy")
PIL = import("PIL")
cv2 = import("cv2")

img = cv2$imread('svm.png')

tl$configure("runs/run-1", flush_secs=5)

v1 = list(0.05,0.11,0.22,0.33,0.44,0.55,0.66,0.77,0.88,0.99)
v2 = list(0.76, 0.55, 0.52, 0.48, 0.46, 0.40, 0.33, 0.31, 0.28, 0.22)
bins = list(0, 1, 2, 3,4,5)

for (step in 1:10){ 
  tl$log_value('accuracy', v1[[step]], step)
  tl$log_value('mse', v2[[step]], step)
}
arr = np$random$randint(10)
tl$log_histogram('histogram',c(bins, list(0.9, 1.1, 2.6, 3.3, 4.7), arr))
tl$log_images('Model',list(img))
tensorboard("runs/run-1")

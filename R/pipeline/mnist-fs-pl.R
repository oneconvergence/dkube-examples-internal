library("reticulate")
library("jsonlite")
kfp <- import("kfp")
json <-import("json")
dsl <- kfp$dsl
compiler <- kfp$compiler

components_url <- "https://raw.githubusercontent.com/oneconvergence/dkube/master/components/"
dkube_preprocessing_op  <- kfp$components$load_component_from_url(paste0(components_url,"preprocess/component.yaml"))
dkube_training_op       <- kfp$components$load_component_from_url(paste0(components_url, "training/component.yaml"))
dkube_serving_op        <- kfp$components$load_component_from_url(paste0(components_url, "serving/component.yaml"))

token <- Sys.getenv("DKUBE_USER_ACCESS_TOKEN")
training_program <- "mnist-fs"
preprocessing_dataset <- "mnist"
training_featureset <- "mnist-fs"
training_output_model <- "mnist"
t_image <- "docker.io/ocdr/dkube-datascience-tf-cpu:v2.0.0"
data_preprocess_script <- "python featureset.py --fs mnist-fs"
framework <- "tensorflow"
version <- "2.0.0"
data_preprocess_input_mounts <- toJSON("/opt/dkube/input/")
featureset_preprocess_output_mounts <- toJSON("/opt/dkube/output/")
training_container <- paste0("{","\"image\": \"",t_image,"\"}")
training_script <- "python model.py --fs mnist-fs"
training_input_featureset_mount <- toJSON("/opt/dkube/input/")
training_output_mount <- toJSON("/opt/dkube/output/")
serving_device <- 'cpu'
serving_image <- "ocdr/tensorflowserver:2.0"
serving_image <- paste0("{","\"image\":\"",serving_image,"\"}")
transformer_image <- paste0("{","\"image\": \"",t_image,"\"}")
transformer_code <- "tf/classification/mnist-fs/digits/transformer/transformer.py"

mnist_fs_pipeline <- function(token = token, dumy = "Empty"){
  preprocess = dkube_preprocessing_op(token, training_container,
                            program=training_program, run_script=data_preprocess_script,
                            datasets=toJSON(preprocessing_dataset),output_featuresets=toJSON(training_featureset),
                            input_dataset_mounts=data_preprocess_input_mounts, output_featureset_mounts=featureset_preprocess_output_mounts)
  train = dkube_training_op(token, training_container,
                            program=training_program, run_script=training_script,
                            featuresets=toJSON(training_featureset), outputs=toJSON(training_output_mount),
                            input_featureset_mounts=training_input_featureset_mount, output_mounts=training_output_mount,
                            framework=framework, version=version)$after(preprocess)
  serving <- dkube_serving_op(token, train$outputs[['artifact']],
                            device=serving_device, serving_image=serving_image,
                            transformer_image=transformer_image,
                            transformer_project=training_program,
                            transformer_code=transformer_code)$after(train)
}
compiler$Compiler()$compile(r_to_py(mnist_fs_pipeline), 'dkube_mnist_fs.tar.gz') 

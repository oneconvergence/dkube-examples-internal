# RESNETv2 network classifier
This example is derived from [retraining](https://www.tensorflow.org/hub/tutorials/image_retraining) and modified to run on Dkube Platform. The example retrains a [resnetv2](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1) model for binary classification with catsdogs data.

 - This program trains RESNETV2 network on CatsDogs data.
 - Modified program is configurable and takes Hyperparameters like steps, batchsize, learning rate etc from ENV vars or from a JSON file. User can input these parameters from Dkube UI or upload a file which will then be provided for the running instance of program.
 - Program is modified to be able to run in distributed training mode. User can select this mode while training in Dkube.
 - Program is modified to write TF summaries and use custom Dkube session hook to emit out some critical information about the training which is then displayed on Dkube UI Dashboard.

# Directories

 - **classifier/program** : This directory has training code files implemented on top of Tensorflow framework.
 - **classifier/data**: This directory has processed training data. Program trains on this data.
 - **inference**: This directory has compatible test data images which can be used for inference.
 - **hptuning/tuning.yaml**: Sample YAML showing the configuration format and parameters for tuning.
 - **pipeline/dkube-resnet-pipeline.ipynb**: Ipython notebook with the code. Upload the file in Dkube notebook and run all the cells. This notebook will generate and trigger the run of pipeline.

# How to Train
## Step1: Create a Code

1. Click Repos side menu option.
2. Click *+Code* button.
3. Enter a name say resnet-catsdogs
4. Enter tag name in Tag text-box
5. Select Source as Git
6. Paste link https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/resnet/catsdogs/classifier/program in the URL text box
7. Enter branch name or version in Branch text-box
8. Click *Add Code* button.
9. Code will be created and imported in Dkube. Progress of import can be seen.
10. Please wait till status turns to ready.


## Step2: Create a dataset
1. Click +Datasets button under Datasets section.
2. Enter a unique name say resnet-catsdogs
3. Enter tag name in Tag text-box, this field is optional.
4. Select Versioning as DVS 
5. Select Dataset store as default
6. Select Dataset Source as Git
7. Paste link https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/resnet/catsdogs/classifier/data in the URL text box
8. Enter branch name or version  in Branch text-box
9. Click the Add Dataset button.
10. Dataset will be created and imported in Dkube. Progress of import can be seen.
11. Please wait till status turns to ready.

## Step-3 Create a Model

This step is to create a dvs model which will hold the trained output model.

1. Click Repos side menu option.
2. Click +Model button under Models.
3. Enter a unique name say catsdogs
4. Click the Add Model button.
5. Model will be created on Dkube.
6. Please wait till status turns to ready

## Step4: Start a training job
1. Click Runs side menu option.
2. Click +Run and select the Training button.
3. Fill the fields in Job form and click the Submit button. See below for sample values to be given in the form, for advanced usage please refer to Dkube User Guide.
   - **Basic Tab**
      - Enter Unique name say Training-catsdogs
      - Code: Click on + button and select resnet-catsdogs project
      - Container Section
        - Select Framework as tensorflow
        - Select Framework version 1.14
        - Start-up command : python model.py
        - Click Next
   - **Repos Tab**
      - Dataset: Under Inputs section, click on + button and select catsdogs dataset and enter mount path say /opt/dkube/input
      - Models: Under Outputs section,click on + button and select catsdogs model and enter mount path: Enter path say /opt/dkube/output
      - Click Next
   - **Configuration Tab**
      - Enter GPUs in GPUs to allocate text-box
4. Click on Submit Button
5. A new entry with name catsdogs-classifier will be created in Jobs table
6. Check the Status field for the lifecycle of the job, wait till it shows complete.


## How to Serve
After the job is complete from above step. The trained model will get generated inside Dkube.

1. Click on the training run named *run-training-Catsdogs.
2. Select Lineage tab beside the Summary details page.
3. Click on OUTPUTS model Catsdogs
4. Click on the Test Inference button.
5. Input the unique name say catsdogs-serving
6. Select CPU or GPU to deploy the model on a specific device. Unless specifically required, model can be served on CPU
7. Click on Test inference button
8. Click Test Inferences in side menu under WORKFLOW section
9. Wait till status field shows running
10. Copy the URL shown in Endpoint field of the serving job.

## Test Inference Details
1. Serving image:  (use default one)
2. Transformer image: (use default one)
3. Transformer Project : (use default one)
4. Transformer code: tf/classification/resnet/catsdogs/transformer/transformer.py

## How to test Inference
1. To test inference open a new tab with link < DKUBE_URL:port/inference >
2. Copy and Paste Dkube OAuth token from Developer Settings present in menu on top right to Authorization
3. Select Model type as Catsdogs
4. Upload an image for inference, images in the inp_samples folder can be used
5. Click predict button and a chart is displayed with probabilities returned by the model

# Hyperparameter tuning
1. Hyperparameter tuning is useful to find the appropriate parameter space for DL training. Dkube will auto generate all the possible combinations of parameters specified and runs training for each of the combination till the goal specified or max count is reached.
2. Dkube plots the graphs for comparision and suggests a best run with hyperparameters used for the run.
3. Create a job same as explained in section  *[\[How to Train\]](#How%20to%20Train)* except that now a tuning file also needs to be uploaded in the *Parameters Tuning* section of the *Training Job*  form.
4. For this example, sample tuning file is present in the *github* as explained in section [Directories](#%20Directories). Alternately, showing the content below - copy-paste and create a new file for upload.
```
parallelTrialCount: 3
maxTrialCount: 6
maxFailedTrialCount: 3
objective:
  type: maximize
  goal: 0.99
  objectiveMetricName: accuracy
algorithm:
  algorithmName: random
parameters:
  - name: --learning_rate
    parameterType: double
    feasibleSpace:
      min: "0.001"
      max: "0.005"
  - name: --batch_size
    parameterType: int
    feasibleSpace:
      min: "100"
      max: "200"


```
5. Upload this file and click *Submit* button.

## Pipeline
1. Training, Serving & Inference stages explained in above sections can be automated using Dkube pipeline.
2. Sample pipeline for this example is available under `pipeline/` mentioned in section [directories](#%20Directories).

## How to use dkube-resnet-pipeline.ipynb
1. Create Code with name catsdogs
2. Create Dataset with name catsdogs
3. Create model with name catsdogs
4. Go to Default Dkube notebook
5. Then, click the Jupyter icon which will open a UI. Select the dkube-resnet-pipeline.ipynb under pipeline/ipynbs and double click it to open
6. Fill training_program, training_dataset, training_output_model with proper values
7. Run all the cells of dkube-resnet-pipeline.ipynb This will create a pipeline, creates an experiment and a run.
8. Links are displayed in the output cells wherever applicable.

## How to use dkube_resnetv2_pl.tar.gz
1. Click *Pipelines* sidemenu option.
2. Click *+Upload pipeline* and upload this file.
3. Click *Experiments* sidemenu option.
4. Click *Create an experiment* button and input a unique *experiment* name.
5. Click *next* button it will auto display form to create a new *run*.
6. Select the *pipeline* which was uploaded in *step 2*
7. Fill in the *Run Parameters* fields.

## Release Model
1. Go to the model version and click the Release Model icon.
2. Once the model is released, it will be available in the Released view of the Models.

## Publish Model
1. A model can be published directly from the repo or can be first released and then published.
2. Go to the model version and click Publish Model icon.
3. Give a name and check the transformer.
   - Edit transformer code field and replace the default test with tf/classification/resnet/catsdogs/transformer/transformer.py
   - Click Submit.
4. Once a model is published, it will be available in the Model Catalog.


## Model Serving
1.  A published model can be staged or deployed from the Model Catalog.
2.  Go to Model Catalog and click on the published model.
3.  Go to the model version and click stage/deploy.
4.  Give a name and choose CPU and submit.
5.  Open the Inference UI page https://<IP>:32222/inference
6.  Fill serving URL and auth token values.
7.  Choose model catsdogs
8.  Download test image cat.png from https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/resnet/catsdogs/inp_samples
9.  Upload the downloaded file
10. Click Predict.

## Steps for running the program in IDE
1. Create a IDE with tensorflow framework and version 1.14.
2. Select the Code resnet-catsdogs.
3. Under Inputs section, in Repos Tab select the dataset resnet-catsdogs and enter mount path /opt/dkube/input.
4. Create a new notebook inside workspace/resnet-catsdogs/tf/classification/resnet/catsdogs/classifier/program
   - In a cell type %load model.py in a notebook cell and then run.

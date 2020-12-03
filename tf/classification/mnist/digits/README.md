# MNIST network classifier
This example is derived from [tensorflow example](https://github.com/tensorflow/models/tree/r1.4.0/official/mnist) and modified to run on Dkube Platform.

 - This program trains MNIST network on Gray scale Digits data.
 - Modified program is configurable and takes Hyperparameters like steps, batchsize, learning rate etc from ENV vars or from a JSON file. User can input these parameters from Dkube UI or upload a file which will then be provided for the running instance of program.
 - Program is modified to be able to run in distributed training mode. User can select this mode while training in Dkube.
 - Program is modified to write TF summaries and use custom Dkube session hook to emit out some critical information about the training which is then displayed on Dkube UI Dashboard.

# Directories

 - **classifier/program** : This directory has training code files implemented on top of Tensorflow framework.
 - **classifier/data**: This directory has processed training data. Program trains on this data.
 - **inference**: This directory has compatible test data images which can be used for inference.
 - **hptuning/tuning.yaml**: Sample YAML showing the configuration format and parameters for tuning.
 - **pipeline/digits.py**: Python DSL defining a sample Dkube pipeline. Pipeline uses Dkube components for all stages.
 - **pipeline/digits.tar.gz**: Compiled python DSL which can be directly uploaded on to Dkube platform.
 - **pipeline/digits.ipynb**: Ipython notebook with the code. Upload the file in Dkube notebook and run all the cells. This notebook will generate and trigger the run of pipeline.

# How to Train
## Step1: Create a Project

1. Click Repos side menu option.
2. Click +Project button under Projects section.
3. Enter a name say mnist-digits
4. Enter tag name in Tag text-box
5. Select Project Source as Git
6. Paste link https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/classifier/program in the URL text box for tensorflow version 1.14 or https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/classifier/program-2.x for tensorflow version 2.0.
7. Enter branch name or version in Branch text-box.
8. Click the Add Project button.
9. Project will be created and imported in Dkube. Progress of import can be seen.
10. Please wait till status turns to ready.


## Step2: Create a dataset
1. Click Repos side menu option.
2. Click +Datasets button under Datasets section.
3. Enter a unique name say mnist-digits
4. Enter tag name in Tag text-box and field is optional
5. Select Versioning as DVS 
6. Select Dataset store as default
7. Select Dataset Source as Git
8. Paste link https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/classifier/data  in the URL text box.
9. Enter branch name or version  in Branch text-box 
10. Click the Add Dataset button..
11. Dataset will be created and imported in Dkube. Progress of import can be seen.
12. Please wait till status turns to ready.

## Step3: Create a Model
1. Click Repos side menu option.
2. Click +Model button under Models.
3. Enter a unique name say mnist-digits
4. Select Versioning as DVS 
5. Select Model store as default
6. Select Model Source as None.
7. Click the Add Model button.
8. Model will be created on Dkube.
9. Please wait till status turns to ready.


## Step4: Start a training job
 1. Click *Runs* side menu option.
 2. Click *+Run* and then select Training button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
- Basic Tab
  - Enter Unique name say Training-mnist
  - Select Framework as tensorflow
  - Select Framework version 1.14 or 2.0 depending on your choice.
  - Start-up command : python model.py
  - Click Next
- Repos Tab
  - Project: Click on + button and select mnist-digits project
  - Project: Click on + button and select mnist-digits project
  - Dataset: Click on + button and select mnist-digits Dataset and 
  - Enter mount path say /opt/dkube/input
- Output Tab
  - Models: Click on + button and select Models.
  - Enter mount path: Enter path say /opt/dkube/output
  - Click Next
- Configuration Tab
  - Enter GPUs in GPUs to allocate text-box
4. Click on Submit Button
5. A new entry with name mnist-digits will be created in Jobs table
6. Check the Status field for the lifecycle of the job, wait till it shows complete.


# How to Serve

 1. After the job is complete from above step. The trained model will get generated inside Dkube.
 2. Click on the training run named *run-training-mnist-digits.
 3. Select Lineage tab beside the Summary details page.
 4. Click on OUTPUTS model mnist-digits
 5. Click on the Test Inference button.
 6. Input the unique name say mnist-serving
 7. Select CPU or GPU to deploy the model on a specific device. Unless specifically required, model can be served on CPU
 8. Click on Test inference button
 9. Click Test Inferences in side menu.
 10. Wait till status field shows running
 11. Copy the URL shown in Endpoint field of the serving job

# Test Inference Details for tensorflow version 1.14 and 2.0 
1. Serving image : (use default one)
2. Transformer image : (use default one)
3. Transformer Project : (use default one)
4. Transformer code : tf/classification/mnist/digits/transformer/transformer.py

# How to test Inference
1. To test inference open a new tab with link https://< DKUBE_URL:port/inference >
2. Copy and Paste Dkube OAuth token from Developer Settings present in menu on top right to Authorization
3. Select Model type as mnist-digits.
4. Upload an image for inference, images in the inference folder can be used.
5. Click predict button and a chart is displayed with probabilities returned by the model.


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
      min: "0.01"
      max: "0.05"
  - name: --batch_size
    parameterType: int
    feasibleSpace:
      min: "100"
      max: "200"


```
5. Upload this file and click *Submit* button.

## How to Compile Pipeline tar
1. Start the default dkube notebook from the IDE tab.
2. Once running, click the jupyterlab icon to launch jupyterlab.
3. Go to pipeline/ipynbs.
4. Double click on digits.ipynb.
5. Run the notebook and create the tar file.
6. Download the tar file by right clicking on it.
7. Upload the tar file into the DKube pipeline UI.

## How to use digits.ipynb
1. Create project with name mnist.
2. Create Dataset with name mnist.
3. Create model with name mnist.
4. Go to Default Dkube notebook.
5. Then, click the Jupyter icon which will open a UI. Select the digits.ipynb and double click it to open.
6. Fill training_program, training_dataset, training_output_model with proper values
7. Run all the cells of digits.ipynb. This will create a pipeline, creates an experiment and a run.
8. Links are displayed in the output cells wherever applicable.

## How to use digits.tar.gz
1. Click *Pipelines* sidemenu option.
2. Click *+Upload pipeline* and upload this file.
3. Click *Experiments* sidemenu option.
4. Click *Create an experiment* button and input a unique *experiment* name.
5. Click *next* button it will auto display form to create a new *run*.
6. Select the *pipeline* which was uploaded in *step 2*
7. Fill in the *Run Parameters* fields.

## How to use digits.py
1. This DSL definition needs to be compiled first. Following prereqs must be installed.
```
python3.5 or greater
pip install --upgrade "urllib3==1.22" 
pip install https://storage.googleapis.com/ml-pipeline/release/1.0.0/kfp.tar.gz --upgrade
```
2. Use the command below to compile the DSL,
```dsl-compile --py [path/to/python/file] --output [path/to/output/tar.gz]```
3. Once the tar ball is generated. Follow the procedure mentioned in [section](##%20How%20to%20use%20digits.tar.gz). 



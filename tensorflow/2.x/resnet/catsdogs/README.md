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
 - **pipeline/catsdogs.py**: Python DSL defining a sample Dkube pipeline. Pipeline uses Dkube components for all stages.
 - **pipeline/catsdogs.tar.gz**: Compiled python DSL which can be directly uploaded on to Dkube platform.
 - **pipeline/catsdogs.ipynb**: Ipython notebook with the code. Upload the file in Dkube notebook and run all the cells. This notebook will generate and trigger the run of pipeline.

# How to Train
## Step1: Create a workspace

 1. Click *Workspaces* side menu option.
 2. Click *+Workspace* button.
 3. Select *Github* option.
 4. Enter a unique name say *resnet-catsdogs*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/classification/resnet/catsdogs/classifier/program](https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/classification/resnet/catsdogs/classifier/program)* in the URL text box.
 6. Click *Add Workspace* button.
 7. Workspace will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

## Step2: Create a dataset
 1. Click *Datasets* side menu option.
 2. Click *+Dataset* button.
 3. Select *Github* option.
 4. Enter a unique name say *resnet-catsdogs*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/classification/resnet/catsdogs/classifier/data](https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/classification/resnet/catsdogs/classifier/data)* in the URL text box.
 6. Click *Add Dataset* button.
 7. Dataset will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.
## Step3: Start a training job
 1. Click *Jobs* side menu option.
 2. Click *+Training Job* button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
	- Enter a unique name say *catsdogs-classifier*
	- **Container** section
		- Tensorflw version - Leave with default options selected.
		- Start-up script -`python model.py`
	- **GPUs** section - Provide the required number of GPUs. This field is optional, if not provided network will train on CPU.
	-  **Parameters** section - Input the values for hyperparameters or leave it to default. This program trains to very good accuracy with the displayed default parameters.
	- **Workspace** section - Please select the workspace *resnet-catsdogs* created in *Step1*.
	- **Model** section - Do not select any model.
	- **Dataset** section - Please select the dataset *resnet-catsdogs* created in *Step1*.
4. Click *Submit* button.
5. A new entry with name *catsdogs-classifier* will be created in *Jobs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

# How to Serve

 1. After the job is *complete* from above step. The trained model will get generated inside *Dkube*. Link to which is reflected in the *Model* field of a job in *Job* table.
 2. Click the link to see the trained model details.
 3. Click the *Deploy* button to deploy the trained model for serving. A form will display.
 4. Input the unique name say *catsdogs-serving*
 5. Select *CPU* or *GPU* to deploy model on specific device. Unless specifically required, model can be served on CPU.
 6. Click *Deploy* button.
 7. Click *Inferences* side menu and check that a serving job is created with the name given i.e, *catsdogs-serving*.
 8. Wait till *status* field shows *running*.
 9. Copy the *URL* shown in *Endpoint* field of the serving job.

# How to test Inference
1. To test inference **dkubectl** binary is needed.
2. Please use *dkube-notebook* for testing inference.
3. Create a file *catsdogs.ini* with below contents, Only field to be filled in is *modelurl*. Paste the *URL* copied in previous step.

    ```
    [INFAPP]
    #Name of the program to run, choices are - digits,catsdogs,cifar,objdetect,bolts
    program="catsdogs"
    #Serving URL of the model in dkube
    modelurl=""
    #Container image to be used for inference app
    #image=""
    #IP to make inference app available on
    #accessip=""
    
    ################################################################################################################
    #    Following fields need not be filled in when launching inference application from inside dkube notebook    #
    ################################################################################################################
    #Name of the dkube user
    #user=""
    #Path to the kubeconfig of the cluster ex: "~/.dkube/kubeconfig"
    #kubeconfig=""
    ```
  4. Execute the command `dkubectl infapp launch --config catsdogs.ini -n catsdogs`
  5. The above command will output a URL, please click the URL to see the UI which can be used for testing inference.
  6. Upload an image for inference, images in **inference** folder can be used.
  7. Click *predict* button and a chart is displayed with probabilities returned by the model.

# Hyperparameter tuning
1. Hyperparameter tuning is useful to find the appropriate parameter space for DL training. Dkube will auto generate all the possible combinations of parameters specified and runs training for each of the combination till the goal specified or max count is reached.
2. Dkube plots the graphs for comparision and suggests a best run with hyperparameters used for the run.
3. Create a job same as explained in section  *[\[How to Train\]](#How%20to%20Train)* except that now a tuning file also needs to be uploaded in the *Parameters Tuning* section of the *Training Job*  form.
4. For this example, sample tuning file is present in the *github* as explained in section [Directories](#%20Directories). Alternately, showing the content below - copy-paste and create a new file for upload.
```
studyName: tfjob-example
owner: crd
optimizationtype: maximize
objectivevaluename: train_accuracy_1
optimizationgoal: 0.99
requestcount: 1
metricsnames:
 - train_accuracy_1
parameterconfigs:
 - name: --learning_rate
   parametertype: double
   feasible:
     min: "0.01"
     max: "0.05"
 - name: --batch_size
   parametertype: int
   feasible:
     min: "100"
     max: "200"
 - name: --num_epochs
   parametertype: int
   feasible:
     min: "1"
     max: "10"
```
5. Upload this file and click *Submit* button.
# Pipeline
1. Training, Serving & Inference stages explained in above sections can be automated using Dkube pipeline.
2. Sample pipeline for this example is available under `pipeline/` mentioned in section [directories](#%20Directories)
## How to use catsdogs.ipynb
1. Create a new *Workspace* in Dkube as explained in [section](##%20Step1:%20Create%20a%20workspace) with unqiue name say, *catsdogs-pl-nb*
2. Change the github url to [https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/classification/resnet/catsdogs/pipeline/catsdogs.ipynb](https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/classification/resnet/catsdogs/pipeline/catsdogs.ipynb)
3. Create a new *Notebook* in Dkube and select the workspace as *catsdogs-pl-nb*.
4. The Dkube notebook does not need any Dataset.
5. Launch the notebook and wait for the status to show *running*.
6. Then, click the *Jupyter* icon which will open a UI. Selec the *catsdogs.ipynb* and double click it to open.
7. Run all the cells of *catsdogs.ipynb*. This will create a pipeline, creates an experiment and a run.
8. Links are displayed in the output cells wherever applicable.
## How to use catsdogs.tar.gz
1. Click *Pipelines* sidemenu option.
2. Click *+Upload pipeline* and upload this file.
3. Click *Experiments* sidemenu option.
4. Click *Create an experiment* button and input a unique *experiment* name.
5. Click *next* button it will auto display form to create a new *run*.
6. Select the *pipeline* which was uploaded in *step 1*
7. Fill in the *Run Parameters* fields. Meaning of each of the field is explained here -> [Dkube Components](https://github.com/oneconvergence/gpuaas/tree/dkube_1.4.1_release/dkube/pipeline/components)
## How to use catsdogs.py
1. This DSL definition needs to be compiled first. Following prereqs must be installed.
```
python3.5 or greater
pip install --upgrade "urllib3==1.22" 
pip install https://storage.googleapis.com/ml-pipeline/release/0.1.18/kfp.tar.gz --upgrade
```
2. Use the command below to compile the DSL,
```dsl-compile --py [path/to/python/file] --output [path/to/output/tar.gz]```
3. Once the tar ball is generated. Follow the procedure mentioned in [section](##%20How%20to%20use%20catsdogs.tar.gz). 

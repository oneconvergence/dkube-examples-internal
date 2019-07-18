# MNIST network classifier
This example is derived from [keras example](https://github.com/keras-team/keras/blob/master/examples/mnist_transfer_cnn.py) and modified to run on Dkube Platform.

 - This program trains a simple convnet on the MNIST dataset the first 5 digits [0..4].
 - Freezes convolutional layers and fine-tunes dense layers for the classification of digits [5..9].
 - Modified program is configurable and takes Hyperparameters like epochs, batchsize etc from ENV vars. User can input these parameters from Dkube UI which will then be provided for the running instance of program.
 - Modified the program to convert the keras h5 model format to tensorflow savedmodel.pb format.

# Directories

 - **classifier/program** : This directory has training code files implemented on top of keras framework.
 - **classifier/docker**: This directory has the docker file used to build docker image for custom training.
 - **inference/server**: This directory has the custom inference server code.
 - **inference/docker**: This directory has the docker file used to build docker image for custom inference.
 - **inference/images**: This directory has compatible test data images which can be used for inference.

# How to Train
## Step1: Create custom docker image for training
1. git clone https://github.com/oneconvergence/dkube-examples.git -b keras_custom_training
2. cd dkube-examples/keras/classification/mnist/classifier
3. sudo docker build -t ocdr/custom-datascience-keras:training-gpu -f docker/custom-datascience-keras-gpu.dockerfile .
4. sudo docker push ocdr/custom-datascience-keras:training-gpu

## Step2: Create a workspace
 1. Click *Workspaces* side menu option.
 2. Click *+Workspace* button.
 3. Select *Github* option.
 4. Enter a unique name say *keras-mnist*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/keras_custom_training/keras/classification/mnist/classifier/program](https://github.com/oneconvergence/dkube-examples/tree/keras_custom_training/keras/classification/mnist/classifier/program)* in the URL text box.
 6. Click *Add Workspace* button.
 7. Workspace will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

## Step3: Start a training job
 1. Click *Jobs* side menu option.
 2. Click *+Training Job* button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
	- Enter a unique name say *keras-digits-classifier*
	- **Container** section
		- Select Container - Select Custom
		- Docker Image URL - ocdr/custom-datascience-keras:training-gpu created in *Step1*.
		- Private - Fill in username and password if docker image is private or else leave private disabled.
		- Start-up script -`python model.py`
	- **GPUs** section - Provide the required number of GPUs. This field is optional, if not provided network will train on CPU.
	-  **Hyper parameters** section - Input the values for hyperparameters. Use following values for better result:
		- Epochs - 5
		- Batch size - 128
		- Steps - 100
	- **Workspace** section - Please select the workspace *keras-mnist* created in *Step2*.
	- **Model** section - Do not select any model.
	- **Dataset** section - Do not select any model.
4. Click *Submit* button.
5. A new entry with name *digits-classifier* will be created in *Jobs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

# How to Serve

 1. After the job is *complete* from above step. The trained model will get generated inside *Dkube*. Link to which is reflected in the *Model* field of a job in *Job* table.
 2. Click the link to see the trained model details.
 3. Click the *Deploy* button to deploy the trained model for serving. A form will display.
 4. Input the unique name say *keras-digits-serving*
 5. Select *CPU* or *GPU* to deploy model on specific device. Unless specifically required, model can be served on CPU.
 6. Click *Deploy* button.
 7. Click *Inferences* side menu and check that a serving job is created with the name given i.e, *keras-digits-serving*.
 8. Wait till *status* field shows *running*.
 9. Copy the *URL* shown in *Endpoint* field of the serving job.

# How to test Inference
## Step1: Create custom docker image for inference
1. git clone https://github.com/oneconvergence/dkube-examples.git -b keras_custom_training
2. cd dkube-examples/keras/classification/mnist/inference
3. sudo docker build -t ocdr/custom-datascience-keras:inf-mnist -f docker/custom-inference-server.dockerfile .
4. sudo docker push ocdr/custom-datascience-keras:inf-mnist

## Step2: Create custom job
1. To test inference **dkubectl** binary is needed.
2. Please use *dkube-notebook* for testing inference. We need to create a custom job and deploy a flask server to talk to the tensorflow model server running in dkube.
3. Create a file *custom.ini* with below contents.

    ```
    [CUSTOM_JOB]
    #URL at which dkube is available - https://ip:port
    dkubeURL=""
    #JWT token to access dkube APIs
    token=""
    #Name of the custom job
    name="custom-inf-server"
    #Container image to be used for the job POD(Format: registry/repo/image:[tag])
    image="ocdr/custom-datascience-keras:inf-mnist"
    #Tags for the custom job
    tags=[]
    #Dkube workspace
    workspace=""
    #Startup script to run the program on launching the job
    script="python server.py <Paste the URL copied in previous step *How to Serve*>"
    #Datasets to be used for the job
    datasets=[]
    #Models to be used for the job
    models=[]
    #Environment variables to be set in the container ["key:value"]
    envs=[]
    #Docker username(If private repository)
    dockerusername=""
    #Docker password(If private repository)
    dockerpassword=""
    #Targetport of service(If job is services)
    targetport=5000
    #Choice of exposing the service. One of dkubeproxy or nodeport(If job is services)
    exposeas="nodeport"
    ```
  4. Execute the command `dkubectl customjob start --config custom.ini`
  5. The above command will output a URL. Copy the URL.
  
  ## Step3: Run Client
  1. Replace the *public ip* with dkube public ip in the above URL.
  2. Run the below curl command by replacing the *server url* with url obtained in previous step and *image path* with path to test image.  
	`curl <server url>/predict --request POST -F "file=@<image path>"`

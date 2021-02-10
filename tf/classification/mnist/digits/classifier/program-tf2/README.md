## Tensorflow 2 Example (Without Compat mode)
# How to Train 
## Step1: Create a Code

1. Click Repos side menu option.
2. Click +Code button.
3. Enter a name say mnist-digits
4. Enter tag name in Tag text-box
5. Select Source as Git
6. Paste link https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist/digits/classifier/program-tf2 in the URL text box for tensorflow version 2.0
7. Enter branch name or version in Branch text-box.
8. Click the Add Code button.
9. Code will be created and imported in Dkube. Progress of import can be seen.
10. Please wait till status turns to ready.


## Step2: Create a dataset
1. Click Repos side menu option.
2. Click +Datasets button under Datasets section.
3. Enter a unique name say mnist-digits
4. Enter tag name in Tag text-box and field is optional
5. Select Versioning as DVS 
6. Select Dataset store as default
7. Select Dataset Source as Git
8. Paste link https://github.com/oneconvergence/dkube-examples-internal/tree/master/pytorch/classification/mnist/digits/classifier/data/MNIST/raw in the URL text box.
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
    - **Basic Tab**
      - Enter Unique name say Training-mnist
      - Code: Click on + button and select mnist-digits code.
      - Container Section
        - Select Framework as tensorflow
        - Select Framework version 2.0.
        - Start-up command : python model.py
        - Click Next
    - **Repos Tab**
      - Dataset: Under Inputs section, click on + button and select mnist-digits Dataset and enter mount path say /opt/dkube/input
      - Models: Under Outputs section,click on + button and select mnist-digits model and enter mount path: Enter path say /opt/dkube/output
      - Click Next
   - **Configuration Tab**
      - Enter GPUs in GPUs to allocate text-box
4. Click on Submit Button
5. A new entry with name mnist-digits will be created in Jobs table
6. Check the Status field for the lifecycle of the job, wait till it shows complete.

## Step 5: Inference Details
1. Serving image : (use default one)
2. Transformer image : (use default one)
3. Transformer Project : (use default one)
4. Transformer code : tf/classification/mnist/digits/classifier/program-tf2/transformer/transformer.py

## How to test inference
1. To test inference open a new tab with link https://< DKUBE_URL:port/inference >
2. Copy and Paste Dkube OAuth token from Developer Settings present in menu on top right to Authorization
3. Select Model type as mnist-digits.
4. Upload an image for inference, images in the inference folder can be used.
5. Click predict button and a chart is displayed with probabilities returned by the model.

## Hyperparameter Tuning
1. Hyperparameter tuning is useful to find the appropriate parameter space for DL training. Dkube will auto generate all the possible combinations of parameters specified and runs training for each of the combination till the goal specified or max count is reached.
2. Dkube plots the graphs for comparision and suggests a best run with hyperparameters used for the run.
3. Create a job same as explained in section [How to Train] except that now a tuning file also needs to be uploaded in the configuration tab under Parameters of the Training Job form.
4. For this example, sample tuning file is present in the github at https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist/digits/hptuning (tuning.json or tuning.yaml).These files can be modified according to the need.

## Running the example in IDE
1. Create a IDE with tensorflow framework and version 2.0.
2. Select the code  mnist-digits.
3. In Repos Tab, Under Inputs section select the mnist-digits dataset and enter the mount path /opt/dkube/input.
4. Create a new notebook inside workspace/program-tf2/program/ and type %load model.py in a notebook cell and then run.
5. Model will be trained and will be available under /opt/dkube/output/1, can be verified by using command !ls /opt/dkube/output/1.


## Tensorflow 2.3 Advanced Example
## Step1: Create a Code
1. Click Repos side menu option.
2. Click +Code button.
3. Enter a name say mnist-digits
4. Enter tag name in Tag text-box
5. Select Source as Git
6. Paste link https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist/digits/classifier/program-tf2 in the URL text box for tensorflow version 2.0
7. Enter branch name or version in Branch text-box.
8. Click the Add Code button.
9. Code will be created and imported in Dkube. Progress of import can be seen.
10. Please wait till status turns to ready.

## Step2 : Create a Dataset
1. Click Repos side menu option.
2. Click +Datasets button under Datasets section.
3. Enter a unique name say mnist-digits
4. Select Versioning as DVS
5. Select Dataset store as default
6. Select Dataset Source as Other.
7. Paste link https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz in the URL text box.
8. Click the Add Dataset button..
9. Dataset will be created and imported in Dkube. Progress of import can be seen.
10.Please wait till status turns to ready.

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
    - **Basic Tab**
      - Enter Unique name say Training-mnist
      - Code: Click on + button and select mnist-digits code.
      - Container Section
        - Select Framework as tensorflow
        - Select Framework version 2.3
        - Start-up command : python model_advanced.py
        - Click Next
    - **Repos Tab**
      - Dataset: Under Inputs section, click on + button and select mnist-digits Dataset and enter mount path say /opt/dkube/input
      - Models: Under Outputs section,click on + button and select mnist-digits model and enter mount path: Enter path say /opt/dkube/output
      - Click Next
   - **Configuration Tab**
      - Enter GPUs in GPUs to allocate text-box
4. Click on Submit Button
5. A new entry with name mnist-digits will be created in Jobs table
6. Check the Status field for the lifecycle of the job, wait till it shows complete.

## Step5: Serving, inference (without a transformer, only CURL)
1. Click on Repos tab, navigate to models and then go to mnist-digits model and select version and create a test inference. Don't check transformer.
2. Download the inp data from https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/tf/classification/mnist/digits/classifier/program-tf2/sample.json
3. Go to your test inference, copy curl command and add  -kv after curl, remove --data-raw till the end and append -d @sample.json at the end.

## Running the example in IDE
1. Create a IDE with tensorflow framework and version 2.3
2. Select the code  mnist-digits.
3. In Repos Tab, Under Inputs section select the mnist-digits dataset and enter the mount path /opt/dkube/input.
4. Inside workspace/program-tf2/ folder, run all the cells of tf-advanced.ipynb.


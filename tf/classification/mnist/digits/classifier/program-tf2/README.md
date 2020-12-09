# How to Train 
## Step1: Create a Project

1. Click Repos side menu option.
2. Click +Project button under Projects section.
3. Enter a name say mnist-digits
4. Enter tag name in Tag text-box
5. Select Project Source as Git
6. Paste link https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/classifier/program-tf2 in the URL text box for tensorflow version 2.0
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
8. Paste link https://github.com/oneconvergence/dkube-examples/tree/master/pytorch/classification/mnist/digits/classifier/data/MNIST/raw in the URL text box.
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
      - Project: Click on + button and select mnist-digits project
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

## Step 5: Inference is yet to be added.

## Running the example in IDE
1. Create a IDE with tensorflow framework and version 2.0.
2. Select the project mnist-digits.
3. In Repos Tab, Under Inputs section select the mnist-digits dataset and enter the mount path /opt/dkube/input.
4. Create a new notebook inside workspace/program-tf2/program/ and type %load model.py in a notebook cell and then run.
5. Model will be trained and will be available under /opt/dkube/output, can be verified by using command !ls /opt/dkube/output.





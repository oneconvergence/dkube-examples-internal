# Metrics examples
These examples are derived from [mlflow examples](https://github.com/mlflow/mlflow/tree/master/examples) and modified to run on Dkube Platform.

# Directories

 - **sklearn_elasticnet_diabetes** : sklearn examples with log_metrics API
 - **tf2**: tensorflow 2.0 example with autolog API

# How to Train
## Step1: Create a Code

1. Click Repos side menu option.
2. Click *+Code* button.
3. Enter a name say metrics-example
4. Enter tag name in Tag text-box
5. Select Code Source as Git
6. Paste link https://github.com/oneconvergence/dkube-examples-internal/tree/master/metrics in the URL text box
7. Enter branch name or version in Branch text-box.
8. Click *Add Code* button.
9. Code will be created and imported in Dkube. Progress of import can be seen.
10. Please wait till status turns to ready.

## Step2: Create a Model
1. Click Repos side menu option.
2. Click +Model button under Models.
3. Create a model for each example. For eg, metrics-example-skl and metrics-example-tf2
4. Select Versioning as DVS 
5. Select Model store as default
6. Select Model Source as None.
7. Click the Add Model button.
8. Model will be created on Dkube.
9. Please wait till status turns to ready.

## Step3: Start a training job
### Follow the steps below for the sklearn example
 1. Click *Runs* side menu option. 
 2. Click *+Run* and then select Training button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
- **Basic Tab**
  - Enter Unique name say metrics-example-skl
  - Enter Description name in Description text-box
  - Enter Tags name in Tags text-box
  - Code: Click on + button and select metrics-example created in step 1
  - Container Section
    - Select Framework as sklearn
    - Select Version as 0.23.2
    - Start-up command : cd sklearn && python train_diabetes.py
    - Click Next
- **Repos Tab**
    - Models: Under *Outputs* section, Click on + button and select Models.
    - Select metrics-example-skl
    - Enter mount path: Enter path say /opt/dkube/output
4. Click on Submit Button
5. A new entry with name metrics-example-skl will be created in Jobs table
6. Check the Status field for the lifecycle of the job, wait till it shows complete.

### Follow the steps below for the tf2 example
 1. Click *Runs* side menu option. 
 2. Click *+Run* and then select Training button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
- **Basic Tab**
  - Enter Unique name say metrics-example-tf2
  - Enter Description name in Description text-box
  - Enter Tags name in Tags text-box
  - Code: Click on + button and select metrics-example created in step 1.
  - Container Section
    - Select Framework as tensorflow
    - Select Version as 2.0.0
    - Start-up command : cd tf2 && python train_predict_2.py
    - Click Next
- **Repos Tab**
  - Models: Under Outputs section, Click on + button and select Models.
  - Select metrics-example-skl
  - Enter mount path: Enter path say /opt/dkube/output
4. Click on Submit Button
5. A new entry with name metrics-example-skl will be created in Jobs table
6. Check the Status field for the lifecycle of the job, wait till it shows complete.

## Step4: View metrics 
 1. Metrics are visible from each of the Run or the Models
 2. Goto either the metrics-example-sklearn or metrics-example-tf2 Run and click on the Metrics tab
 3. Goto either the metrics-example-sklearn or metrics-example-tf2 Model, select the latest version of the Model and click on the Metics tab

## Steps for running the sklearn metric example in IDE:
1. Create a IDE with sklearn framework and version 0.23.2.
2. Select the code  metrics-example.
3. Create a new notebook inside workspace/metrics-example/metrics/sklearn
   - In first cell type:
     %mkdir -p /opt/dkube/output
     %rm -rf /opt/dkube/output/*
   - In 2nd cell type !python train_diabetes.py 0.01 0.01 in a notebook cell and then run.
4. Note for running the training more than once, please run the cell 1 again.

# MNIST example using PyTorch
## Step1: Create a Project
 1. Click *Repos* side menu option.
 2. Click *+Project* button for Dkube version 2.1.x.x or *+Code* for Dkube version 2.2.x.x.
 3. Select Project source as Git.
 4. Enter a unique name say *mnist-pt*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/pytorch/classification/mnist/digits/classifier/program 
 ](https://github.com/oneconvergence/dkube-examples/tree/master/pytorch/classification/mnist/digits/classifier/program)* in the URL text box.
 6. Click *Add Project* button for Dkube version 2.1.x.x or *Add Code* for Dkube version 2.2.x.x.
 7. Project will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

## Step2: Create a dataset
 1. Click *Datasets* side menu option.
 2. Click *+Dataset* button.
 3. Select *Github* option.
 4. Enter a unique name say *mnist-pt*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/pytorch/classification/mnist/digits/classifier/data
 ](https://github.com/oneconvergence/dkube-examples/tree/master/pytorch/classification/mnist/digits/classifier/data)* in the URL text box.
 6. Click *Add Dataset* button.
 7. Dataset will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

## Step3: Create a model
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Enter a unique name say *mnist-pt*.
 4. Select Versioning as DVS. 
 5. Select Model store as default.
 6. Select Model Source as None.
 7. Click the Add Model button.
 8. Model will be created on Dkube.
 9. Please wait till status turns to ready.


## Step4: Start a training job
 1. Click *Runs* side menu option.
 2. Click *+Runs* and select Training button.
 3. Fill the fields in Run form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - **Basic Tab** :
	  - Enter a unique name say *digits-classifier*
 	  - **Container** section
		- Framework - pytorch.
		- Project section - Please select the workspace *mnist-pt* created in **Step1**.
		- Start-up script -`python model.py`
    - **Repos Tab**
	    - Dataset section - Under Inputs section,select the dataset *mnist-pt* created in **Step2**. Mount point: /opt/dkube/input .
	    - Model section   - Under Outputs section,select the model *mnist-pt* under Outputs created in **Step3**. Mount point: /opt/dkube/output .
4. Click *Submit* button.
5. A new entry with name *digits-classifier* will be created in *Runs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

## Test Inference Details
1. Serving image : (use default one)
2. Transformer image : (use default)
3. Transformer project (use default)
4. Transformer code :pytorch/classification/mnist/digits/classifier/transformer/transformer.py 

## Step5: Create a Test Inference
1. Go to the model version and create inference.
2. Check transformer.
3. Change transformer code to pytorch/classification/mnist/digits/classifier/transformer/transformer.py.
4. Check CPU and create.

## Step6: Inference
1. Download data sample from *[https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/inference](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/inference)*
2. Open the URL https://<set-up-IP>:32222/inference.
3. Copy the serving endpoint from the test inference tab and paste it into the serving the URL field.
4. Copy token from developer settings and paste into token field.
5. Select model mnist from the dropdown.
6. Upload the downloaded image and click predict. 

## Step7: Release, Publish and Stage or Deploy model

1. *Release Model*
- Click on model name mnist-pt.
- Click on Release Action button for latest version of Model.
- Click on Release button, the release will change to released.
2. *Publish Model*
- Click on Publish Model icon under ACTIONS column.
- Give the publish model name.
- Click on Transformer checkbox.
- Change transformer code to pytorch/classification/mnist/digits/classifier/transformer/transformer.py.
- Check CPU and click on Submit.
3. *Stage Model*
- Go to Model catalog and click on published model.
- Click on the stage icon under ACTIONS column.
- Enter stage model name and click on CPU and Submit.
- Model changes to STATE as staged.
- Check in Model serving tab the staged model appears and wait for the status to running.
- Staged model can be used to test the prediction.
4. *Deploy Model*
- Click on Model catalog and select the published model.
- Click on the deploy model icon  under ACTIONS column.
- Enter the deploy model name and select CPU and click Submit.
- The state changes to deployed.
- Check in Model Serving and wait for the deployed model to change to running state.
- Deployed Model can used to test the prediction.
5. *Inference*
- Download data sample from *[https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/inference](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist/digits/inference)*
- Open the URL https://<set-up-IP>:32222/inference.
- Copy the serving endpoint from the Model serving for Staged/Deployed model  and paste it into the serving the URL field.
- Copy token from developer settings and paste into token field.
- Select model mnist from the dropdown.
- Upload the downloaded image and click predict.

## Steps for running the program in IDE
1. Create a IDE with pytorch framework and version 1.6.
2. Select the project mnist-pt.
3. Under Inputs section, in Repos Tab select dataset mnist-pt and enter mount path /opt/dkube/input.
4. Create a new notebook inside workspace/mnist-pt/pytorch/classification/mnist/digits/classifier/program
   - In first cell type:
     - %mkdir -p /opt/dkube/output
     - %rm -rf /opt/dkube/output/*
   - In 2nd cell type %load model.py in a notebook cell and then run.
5. Note for running the training more than once, please run the cell 1 again.

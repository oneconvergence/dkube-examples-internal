# **R-MNIST Digit Classification**
## Step 1: Create a Code
1. Click *Repos* side menu option.
2. Click *+Code* button.
3. Select source as *Git*.
4. Enter a unique name say *r-examples*
5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/R/classification
 ](https://github.com/oneconvergence/dkube-examples/tree/master/R/classification)* in the URL text box.
6. Branch: master
7. Click *Add Code* button.
8. Code will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*

## Step 2: Create a model
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Enter a unique name say *r-mnist*.
 4. Select Versioning as DVS. 
 5. Select Model store as default.
 6. Select Model Source as None.
 7. Click the Add Model button.
 8. Model will be created on Dkube.
 9. Please wait till status turns to ready.

## Step 3: Start a training job
 1. Click *Runs* side menu option.
 2. Click *+Runs* and select Training button.
 3. Fill the fields in Run form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - **Basic Tab**
	   - Enter a unique name say *r-mnist*
	   - Code section - Please select the *r-examples* created in **Step1**.
       - Container section
		 - Framework - Tensorflow
		 - Framework Version - r-2.0.0
		 - Start-up script - Rscript mnist/mnist.R
    -  **Repos Tab**
	    - Model section - Under Outputs section, select the model *r-mnist* created in **Step2**. Mount point: /opt/dkube/model.
4. Click *Submit* button.
5. A new entry with name *r-mnist* will be created in *Runs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

## Model overview: 
 1. Once training is complete you can start the tensorboard from the tensorboard launch icon in the Actions column of jobs. 
 2. The tensorboard will be ready in 30-60 seconds. Then the tensorboard can be launched from the launch tensorboard icon. 

## Serving, inference (without a transformer, only CURL)
 1. Go to the model version from lineage and create inference. Don’t check transformer.
 2. Download the inp data from *[https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/R/classification/mnist/inp_sample/r-mnist-inp.json](https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/R/classification/mnist/inp_sample/r-mnist-inp.json)*
 3. curl -kv -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" MODEL_ENDPOINT -d @r-mnist.json


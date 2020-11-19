## Sonar(RDS model format)
## Step1: Workspace/Project
1. Click *Repos* side menu option.
2. Click *+Project* button.
3. Select source as *Git*.
4. Enter a unique name say *r-examples*
5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/2.1.5/R/classification
 ](https://github.com/oneconvergence/dkube-examples/tree/2.1.5/R/classification)* in the URL text box.
6. Branch: 2.1.5
7. Click *Add Project* button.
8. Project will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*.

## Step2: Create a dataset
 1. Click *Datasets* side menu option.
 2. Click *+Dataset* button.
 3. Select *Github* option.
 4. Enter a unique name say *r-examples*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/2.1.5/R/classification
 ](https://github.com/oneconvergence/dkube-examples/tree/2.1.5/R/classification)* in the URL text box.
 6. Click *Add Dataset* button.
 7. Dataset will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

## Step3: Create a model
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Enter a unique name say *r-sonar*.
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
	- Enter a unique name say *r-examples*
	- **Container** section
		- Framework - custom
		- Image - docker.io/ocdr/dkube-datascience-rs-tf-cpu:v2.0
		- Start-up script - `Rscript sonar/sonar.R`
	- **Project** section - Please select the workspace *r-examples* created in **Step1**.	
	- **Dataset** section - Please select the dataset *r-examples* created in **Step2**. Mount point: /opt/dkube/input .
	- **Model** section - Please select the output model *r-sonar* created in **Step3**. Mount point: /opt/dkube/model .
4. Click *Submit* button.
5. A new entry with name *r-examples* will be created in *Runs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

## Evaluation job:
1. Type:  Training
2. Script: Rscript sonar/sonar-eval.R
3. Project: r-examples
4. Framework: custom
5. Image: docker.io/ocdr/dkube-datascience-rs-tf-cpu:v2.0
6. Input Model: r-sonar
7. Mount-Point: /opt/dkube/model

## Sonar Test Inference(experimental):
1. Go to model r-sonar and create test inference.
2. Use image docker.io/ocdr/custom-kfserving:R as serving the image, don’t check for the transformer.

## Sonar Model Publish(experimental):
1. Go to the model r-sonar repo and click the publish model at the latest version.
2. Use serving image docker.io/ocdr/custom-kfserving:R as serving the image, don’t check for the transformer.








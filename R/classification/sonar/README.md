## Sonar(RDS model format)
## Step1: Create a Project
1. Click *Repos* side menu option.
2. Click *+Project* button for Dkube version 2.1.x.x or *+Code* button for version 2.2.x.x.
3. Select source as *Git*.
4. Enter a unique name say *r-examples*
5. Paste link *[https://github.com/oneconvergence/dkube-examples-internal/tree/master/R/classification
 ](https://github.com/oneconvergence/dkube-examples-internal/tree/master/R/classification)* in the URL text box.
6. Branch: master
7. Click *Add Project* button for Dkube version 2.1.x.x or *Add Code* button for version 2.2.x.x.
8. Project will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*.

## Step2: Create a model
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Enter a unique name say *r-sonar*.
 4. Select Versioning as DVS. 
 5. Select Model store as default.
 6. Select Model Source as None.
 7. Click the Add Model button.
 8. Model will be created on Dkube.
 9. Please wait till status turns to ready.

## Step3: Start a training job
 1. Click *Runs* side menu option.
 2. Click *+Runs* and select Training button.
 3. Fill the fields in Run form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - **Basic Tab**
	  - Enter a unique name say *r-examples*
	  - Project section - Please select the workspace *r-examples* created in **Step1**
      - Container section
		- Framework - Tensorflow
		- Framework version - r-2.0.0
		- Start-up script - `Rscript sonar/sonar.R`
	- **Repos** Tab
	  - Model section - Under the Outputs section,select the model *r-sonar* created in **Step3**. Mount point: **/opt/dkube/model**
4. Click *Submit* button.
5. A new entry with name *r-examples* will be created in *Runs* table
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*

## Evaluation job:
1. Type:  Training
2. Script: `Rscript sonar/sonar-eval.R`
3. Project: r-examples
4. Framework: Tensorflow
5. Framework Version : r-2.0.0
6. Input Model: r-sonar
7. Mount-Point: **/opt/dkube/model**

## Sonar Test Inference:
1. Go to model r-sonar and create test inference.
2. Build the custom serving image from sonar/Dockerflie or use image `docker.io/ocdr/custom-kfserving:R-sonar` as serving the image, don’t check for the transformer.
3. Use sample.json from serving folder and run `curl -kv --location --request POST serving_url --header "Authorization: Bearer TOKEN" --header "Content-Type: application/json"  -d @sample.json`
4. Expected Output: `{"data":{Object":[["R"]]}}`








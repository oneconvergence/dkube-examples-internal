## Stock Prediction using SKlearn and Redshift
To run this example, the user requires to set-up some initials into the Dkube.

## Step1: Workspace/Project
1. Click *Repos* side menu option.
2. Click *+Project* button under the Projects section.
3. Select source as *Git*.
4. Enter a unique name say *stock-prediction*
5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/stock-pred-redshift/model 
 ](https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/stock-pred-redshift/model)* in the URL text box.
6. Branch: master
7. Click *Add Project* button.
8. Project will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*.


## Step2: Create a Dataset
1. First, we need to add the dataset from Redshift.
 - Name: google-stock
 - Versioning: None
 - Source: RedShift
 - Endpoint: http://IP:PORT
 - Database: dkube
 - Username: *
 - Password: *

## Step3: Create a model
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Enter a unique name say *stock-pred*.
 4. Select Versioning as DVS.
 5. Select Model store as default.
 6. Select Model Source as None.
 7. Click the Add Model button.
 8. Model will be created on Dkube.
 9. Please wait till status turns to ready.

## Step4: Start a training job
The job must run in the given sequence, training then evaluation. Tensorboard can be launch after the training job.
1. Click *Runs* side menu option.
 2. Click *+Runs* and select Training button.
 3. Fill the fields in Run form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - Enter a unique name say *stock-pred*
    - **Container** section
        - Framework - sklearn
        - Start-up script -`python train.py`
    - **Project** section - Please select the workspace *stock-prediction* created in **Step1**.
    - **Dataset** section - Please select the dataset *google-stock* created in **Step2**. Mount point: /opt/dkube/input.Select pg_internal version
    - **Model** section - Please select the output model *stock-pred* created in **Step3**. Mount-Point: /opt/dkube/model
4. Click *Submit* button.

## Model overview: 
Once training is complete you can start the tensorboard from the tensorboard launch icon in the Actions column of jobs. 
The tensorboard will be ready in 30-60 seconds. Then the tensorboard can be launched from the launch tensorboard icon. 
In tensorboard, the regression fit of stock data can be seen in the tensorboard image tab. 


## Evaluation job
1. Type: Training
2. Project: stock-prediction
3. Framework: sklearn
4. Script: python evaluation.py
5. Input dataset: google-stock, version: pg_internal
   - Mount-point: /opt/dkube/input
   - Select pg_internal version
7. Input Model: stock-pred
   - Mount-Point: /opt/dkube/model


## Inference
1.  Give the Test Inference name
2.  Click on Transformer checkbox
    - Make sure Serving and Transformer image is sklearn image. It will get filled automatically
    - Select project stock-prediction.	
    - Edit transformer code field and replace the default test with sklearn/stock-pred-redshift/model/transformer.py
    - Choose CPU and Submit. 
3.  Go to Test Inferences page and wait until the Test Inference changes to running state
4.  Copy the Endpoint or serving URL
5.  Open the Inference UI page https://<IP>:32222/inference
6.  Fill serving URL and auth token values.
7.  Choose model sk-stock
8.  Save the text from URL in a CSV file *[https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/sklearn/stock-pred-redshift/dataset/goog.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/sklearn/stock-pred-redshift/dataset/goog.csv)*
9.  Upload the CSV file
10. Click Predict. 

##  Release, Publish and deploy model

1.  Go to the model stock-pred and click on Release model icon under ACTIONS
2.  Click Release button in the Release Model popup
3.  Model goes to released stage
4.  Click on Publish Model icon under ACTIONS column
5.  Give the publish model name
6.  Click on Transformer checkbox
7.  Make sure Serving and Transformer image is sklearn image. It will get filled automatically
    - Select project stock-prediction.
    - Edit transformer code field and replace the default test with sklearn/model/transformer.py
8.  Click on Submit button
9.  Once model is published, go to Model catalog and click on published model
10. Click on the stage icon under ACTIONS column
11. Enter stage model name and click on CPU and Submit
12. Model changes to STATE as staged
13. Check in Model serving tab the staged model appears and wait for the status to running
14. Staged model can be used to test the prediction
15. Click on Model catalog and select the published model
16. Click on the deploy model icon  under ACTIONS column
17. Enter the deploy model name and select CPU and click Submit
18. The state changes to deployed
19. Go to Model Serving and wait for the deployed model to change to running state
20. Copy the URL and open the inference as mentioned from 5th point in Inference section above.


## Notebook Workflow
1.  Launch a new Notebook
2.  Select the default DKube data science image.
3.  Choose the project: stock-prediction
4.  Choose framework: sklearn
5.  Choose dataset:  google-stock
    - Version: pg_internal
    - Mount point: /opt/dkube/input
6.  Submit and wait until the notebook becomes ready.
7.  From the side pane navigate to folder workspace/stock-prediction/sklearn/model
8.  Open workflow-singleDB.ipynb
9.  Change DB_NAME and uncomment cell #r_endpoint, r_database, r_user, r_password
10. The kernel will restart automatically, click ok and run from cell 2


# Stock Prediction using :SKlearn
## Step1: Workspace/Project
1. Click *Repos* side menu option.
2. Click *+Project* button.
3. Select source as *Git*.
4. Enter a unique name say *stock-prediction*
5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/2.1.5/sklearn/model
 ](https://github.com/oneconvergence/dkube-examples/tree/2.1.5/sklearn/model)* in the URL text box.
6. Branch: 2.1.5
7. Click *Add Workspace* button.
8. Workspace will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*.

## Step2: Create a dataset
 1. Click *Datasets* side menu option.
 2. Click *+Dataset* button.
 3. Select *Github* option.
 4. Enter a unique name say *google-stock*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/2.1.5/sklearn/dataset
 ](https://github.com/oneconvergence/dkube-examples/tree/2.1.5/sklearn/dataset)* in the URL text box.
 6. Click *Add Dataset* button.
 7. Dataset will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

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
 1. Click *Runs* side menu option.
 2. Click *+Runs* and select Training button.
 3. Fill the fields in Run form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
	- Enter a unique name say *stock-pred*
	- **Container** section
		- Framework - sklearn.
		- Start-up script -`python train.py`
	- **Project** section - Please select the workspace *stock-prediction* created in **Step1**.	
	- **Dataset** section - Please select the dataset *google-stock* created in **Step2**. Mount point: /opt/dkube/input.
	- **Model** section - Please select the output model *stock-pred* created in **Step3**. Mount-Point: /opt/dkube/model
4. Click *Submit* button.
5. A new entry with name *stock-pred* will be created in *Runs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

## Step5: Model overview
- Once training is complete you can start the tensorboard from the tensorboard launch icon in the Actions column of jobs. 
- The tensorboard will be ready in 30-60 seconds. Then the tensorboard can be launched from the launch tensorboard icon. 
- In tensorboard, the regression fit of stock data can be seen in the tensorboard image tab. 

## Step6: Evaluation
- Type: Training
- Project: stock-prediction
- Framework : sklearn
- Script: python evaluation.py
- Input dataset: google-stock
- Mount-point: /opt/dkube/input
- Input Model: stock-pred
- Mount-Point: /opt/dkube/model

## Test Inference(only for release 2.1.3.1 and later)
1. Go to the model version and click test inference.
2. Give a name and check the transformer.
3. Select project stock-prediction.
4. Edit transformer code field and replace the default test with sklearn/model/transformer.py
5. Choose CPU and Submit. 
6. Open the Inference UI page https://<IP>:32222/inference .
7. Fill serving URL and auth token values.
8. Choose model sk-stock.
9. Save the text from the URL in a CSV file*[https://raw.githubusercontent.com/oneconvergence/dkube-examples/2.1.5/sklearn/dataset/goog.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples/2.1.5/sklearn/dataset/goog.csv)*
10. Upload the CSV file.
11. Click Predict. 

## CURL Example for test inference
curl -kv -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" https://<IP>:32222/dkube/inf/v1/models/d3-inf-3d133119-ec8d-47ce-851f-a7608f4cadd4:predict -d @stock-inp.json

stock-inp.json https://raw.githubusercontent.com/oneconvergence/dkube-examples/2.1.5/sklearn/input_sample/stock-input.json

## Release model
1. Go to the model version and click Release Model icon.
2. Once the model is released, it will be available in the Released view of the Models.

## Publish model
1. A model can be published directly from the repo or can be first released and then published.
2. Go to the model version and click Publish Model icon.
3. Give a name and check the transformer.
4. Edit transformer code field and replace the default test with sklearn/model/transformer.py
5. Click Submit. 
6. Once a model is published, it will be available in the Model Catalog.

## Model Serving
1. A published model can be staged or deployed from the Model Catalog.
2. Go to Model Catalog and click on the published model.
3. Go to the model verison and click stage/deploy.
4. Give a name and choose CPU and submit.
5. Open the Inference UI page https://<IP>:32222/inference
6. Fill serving URL and auth token values.
7. Choose model sk-stock.
8. Save the text from the URL in a CSV file *[https://raw.githubusercontent.com/oneconvergence/dkube-examples/2.1.5/sklearn/dataset/goog.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples/2.1.5/sklearn/dataset/goog.csv)*
9. Upload the CSV file
10. Click Predict.





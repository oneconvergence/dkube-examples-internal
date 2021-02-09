# Stock Prediction using :SKlearn
## Step1: Create a Code
1. Click *Repos* side menu option.
2. Click *+Code* button.
3. Select source as *Git*.
4. Enter a unique name say *stock-prediction*
5. Paste link *[https://github.com/oneconvergence/dkube-examples-internal/tree/master/sklearn/stock-pred/model
 ](https://github.com/oneconvergence/dkube-examples-internal/tree/master/sklearn/stock-pred/model)* in the URL text box.
6. Branch: master
7. Click *Add Code* button.
8. Code will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*.

## Step2: Create a dataset
 1. Click *Datasets* side menu option.
 2. Click *+Dataset* button.
 3. Select *Github* option.
 4. Enter a unique name say *google-stock*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples-internal/tree/master/sklearn/stock-pred/dataset
 ](https://github.com/oneconvergence/dkube-examples-internal/tree/master/sklearn/stock-pred/dataset)* in the URL text box.
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
	- **Basic Tab**
	- Enter a unique name say *stock-pred*
	- Code - Please select the Code *stock-prediction* created in **Step1**.
      - Container section
		- Framework - sklearn.
		- Start-up script -`python train.py`
	- **Repos Tab**
	  - Dataset section - Under Inputs section,select the dataset *google-stock* created in **Step2**. Mount point: /opt/dkube/input.
	  - Model section - Under Outputs section, select the output model *stock-pred* created in **Step3**. Mount-Point: /opt/dkube/model
4. Click *Submit* button.
5. A new entry with name *stock-pred* will be created in *Runs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

## Step5: Model overview
- Once training is complete you can start the tensorboard from the tensorboard launch icon in the Actions column of jobs. 
- The tensorboard will be ready in 30-60 seconds. Then the tensorboard can be launched from the launch tensorboard icon. 
- In tensorboard, the regression fit of stock data can be seen in the tensorboard image tab. 

## Step6: Evaluation
- Type: Training
- Code: stock-prediction
- Framework : sklearn
- Script: python evaluation.py
- Input dataset: google-stock
- Mount-point: /opt/dkube/input
- Input Model: stock-pred
- Mount-Point: /opt/dkube/model

## Test Inference Details
1. Serving image : (use default one)
2. Transformer image : (use default)
3. Transformer project (use default)
4. Transformer code : sklearn/stock-pred/model/transformer.py 

## Test Inference
1. Go to the model version and click test inference.
2. Give a name and check the transformer.
3. Select Code stock-prediction.
4. Edit transformer code field and replace the default test with sklearn/stock-pred/model/transformer.py
5. Choose CPU and Submit. 
6. Open the Inference UI page https://<IP>:32222/inference .
7. Fill serving URL and auth token values.
8. Choose model sk-stock.
9. Save the text from the URL in a CSV file*[https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/stock-pred/dataset/goog.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/stock-pred/dataset/goog.csv)*
10. Upload the CSV file.
11. Click Predict. 

## CURL Example for test inference(works without transformer)
curl -kv -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" https://<IP>:32222/dkube/inf/v1/models/d3-inf-3d133119-ec8d-47ce-851f-a7608f4cadd4:predict -d @stock-inp.json

stock-inp.json https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/stock-pred/input_sample/stock-input.json

## Release model
1. Go to the model version and click Release Model icon.
2. Once the model is released, it will be available in the Released view of the Models.

## Publish model
1. A model can be published directly from the repo or can be first released and then published.
2. Go to the model version and click Publish Model icon.
3. Give a name and check the transformer.
4. Edit transformer code field and replace the default test with sklearn/stock-pred/model/transformer.py
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
8. Save the text from the URL in a CSV file *[https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/stock-pred/dataset/goog.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/stock-pred/dataset/goog.csv)*
9. Upload the CSV file
10. Click Predict.

## Steps for running the program in IDE
1. Create a IDE with sklearn framework and version 0.23.2.
2. Select the Code stock-prediction.
3. Under Inputs section, in Repos Tab select dataset google stock and enter mount path /opt/dkube/input.
4. Create a new notebook inside workspace/stock-pred/sklearn/stock-pred/model
   - In first cell type:
     - %mkdir -p /opt/dkube/output
     - %rm -rf /opt/dkube/output/*
   - In 2nd cell type %load train.py in a notebook cell and then run.
5. Note for running the training more than once, please run the cell 1 again.

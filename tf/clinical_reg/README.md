## Compile file manually

```
a. Start any of the jupyterlab notebook from the IDE tab.
b. Once running, click the jupyterlab icon to launch jupyterlab
c. From any folder
    i. Create a new text file
        1. Copy the content from the link https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/tf/clinical_reg/pipeline/regression_setup.ipynb and paste into the text file,
        2. Save it, and rename the text file to regression.ipynb
d. Open regression.ipynb and run cells to generate the tar file and create run.
e. Download the tar file by right-clicking on it(optional).
f. Upload the tar file into the DKube pipeline UI(optional).

```



# Deploy model.
1. Deploy Model 
   -  Go to Model Catalog and Click on the Published model.
   -  Click on the Deploy Model button. 
   -  Give name. 
   -  Serving image: default 
   -  Deploy using: CPU and Submit. 
   -  Deployed Model will be available in Model Serving.

## Test Inference.

1. Download the data files cli_inp.csv and any sample image from images folder from https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/clinical_reg/inference/data
2. In DKube UI, once the pipeline run has completed, navigate to ‘Test Inferences’ on the left pane
3. Copy the ‘Endpoint’ URL in the row using the clipboard icon
4. Duplicate DKube UI on a new tab and change the URL using the domain name and replacing the remaining path with inference after the domain name. 
   - For e.g, https://URL/inference or  https://1.2.3.4:32222/#/dsinference
5. Enter the following URL into the Model Serving URL box https://dkube-proxy.dkube
6. Copy the token from ‘Developer Settings’ and paste into ‘Authorization Token’ box
7. Select Model Type as ‘Regression’ on the next dropdown selection
8. Click ‘Upload Image’ to load image from [A], ‘Upload File’ to load csv from [A]
9. Click ‘Predict’ to run Inference.

## Regression Notebook Workflow.

1. Go to IDE section
2. Create Notebook 
   - Give a name 
   - Code: regression
   - Datasets: 
         - i.   clinical Mount point: /opt/dkube/input/clinical 
         - ii.  images Mount point: /opt/dkube/input/images 
         - iii. rna Mount Point: /opt/dkube/input/rna
i3. Submit
4. Open workflow.ipynb from location workspace/regression/reg_demo/ 
   - Run cells and wait for output (In case of running the notebook second time, restart the kernel)
5. Delete if workflow.py is already there and export the workflow notebook as executable. 
   - Upload it into Juyterlab. 
   - Make changes in py file, comment/remove the following line numbers: 
        -i. 239-240
        ii. 268 
        iii. 435-end 
  -  Save and commit the workflow.py
6. Create a model named workflow with source none.
7. Create training run using workflow.py 
   - Give a name 
   - Code: regression 
   - Startup command: python workflow.py 
   - Datasets: 
        - i.   clinical Mount point: /opt/dkube/input/clinical 
        - ii.  images Mount point: /opt/dkube/input/images 
        - iii. rna Mount Point: /opt/dkube/input/rna 
   - Output model: workflow, Mount point : /opt/dkube/output

## Compile file manually

```
a. Start the default dkube notebook from the IDE tab.
b. Once running, click the jupyterlab icon to launch jupyterlab
c. Go to the pipeline/components folder
    i. Create a new folder name setup and go inside the folder
   ii. Create a file name component.yaml
  iii. Copy the content from this link https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/tf/clinical_reg/pipeline/component.yaml to the component.yaml file.
d. Go to the pipeline/ipynbs folder
    i. Create a new text file
        1. Copy the content from the link https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/tf/clinical_reg/pipeline/regression_setup.ipynb and paste into the text file,
        2. Save it, and rename the text file to regression.ipynb
e. Run cells to generate the tar file.
f. Download the tar file by right-clicking on it.
g. Upload the tar file into the DKube pipeline UI.

```

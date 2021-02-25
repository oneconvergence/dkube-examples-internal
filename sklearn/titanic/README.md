## **Titanic feature-set example**



1. **Add Code** \
&nbsp;1. Name: titanic \
&nbsp;2. Git URL: [https://github.com/oneconvergence/dkube-examples-internal/tree/master/sklearn/titanic/program](https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/program) 
2. **Add dataset and featureset:** \
&nbsp;1. Dataset Name: titanic \
&nbsp;&nbsp;1.1 Git URL: [https://github.com/oneconvergence/dkube-examples-internal/tree/master/sklearn/titanic/data](https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/data) \
&nbsp;2. Featureset Name: titanic-fs \
&nbsp;&nbsp;2.2 Featurespec upload: none 
3. **Create Model:** \
&nbsp;1. Name: titanic \
&nbsp;2. Source: None
4. **Preprocessing job** \
&nbsp;1. **Type:** pre-processing \
&nbsp;2. **Docker-image:** docker.io/ocdr/d3-datascience-tf-cpu:v1.14 \
&nbsp;3. **Script:** python featureset.py --fs titanic-fs \
&nbsp;&nbsp;3.1. titanic is featureset name, the user can change the name. \
&nbsp;4. **Code:** titanic \
&nbsp;5. **Input dataset:** titanic \
&nbsp;&nbsp;5.1. **Mount-point:** /opt/dkube/input \
&nbsp;6. **Output featureset:** titanic-fs \
&nbsp;&nbsp;6.1. **Mount-Point:** /opt/dkube/output 
5. **Training job** \
&nbsp;1. **Type:** training \
&nbsp;2. **Framework:** sklearn \
&nbsp;3. **Framework Version:** 0.23.2 \
&nbsp;4. **Script:** python model.py --fs titanic-fs \
&nbsp;5. **Project:** titanic \
&nbsp;6. **Input featureset:** titanic-fs \
&nbsp;&nbsp;6.1. **Mount-point:** /opt/dkube/input \
&nbsp;7. **Output model:** titanic \
&nbsp;&nbsp;7.1. **Mount-Point:** /opt/dkube/output
6. **Pipeline run:** \
Download [https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/titanic/pipeline/titanic-pl.ipynb] into DKube Notebook IDE and run all the cells. This auto fills values for pipeline parameters as follows and creates a run under "DKube - Titanic" experiment.\
&nbsp;&nbsp; 6.1 Fills auth token from DKUBE_USER_ACCESS_TOKEN environment variable in Notebook \
&nbsp;&nbsp; 6.2 **training_program:** titanic (This is the code name and should exist) \
&nbsp;&nbsp; 6.3 **preprocessing_dataset:** titanic (This is the dataset name and should exist) \
&nbsp;&nbsp; 6.4 **training_featureset:** titanic-fs  (This is the feature set name and should exist) \
&nbsp;&nbsp; 6.5 **training_output_model:** titanic (This is the model name and should exist)


## **Test Inference:**

1. Go to the model and click test inference,
2. Check transformer option, and replace the transformer script with **sklearn/titanic/program/transformer.py**
3. Choose CPU, and submit.
4. Go to https://&lt;URL>:32222/inference
&nbsp;1. Copy the model serving URL from the test inference tab.
&nbsp;2. Copy the auth token from developer settings, 
&nbsp;3. Select model type sk-stock
&nbsp;4. Copy the contents of [https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/titanic/data/titanic_sample.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/sklearn/titanic/data/titanic_sample.csv) and save then as CSV, and upload.
&nbsp;5. Click predict.

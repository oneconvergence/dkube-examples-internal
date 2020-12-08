## **Titanic feature-set example**



1. **Add Code**
    1. Name: titanic
    2. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/program](https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/program) 
2. **Add dataset and featureset:**
    3. Dataset Name: titanic
        1. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/data](https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/data) 
    4. Featureset Name: titanic-fs
        2. Featurespec upload: none
3. **Create Model:**
    5. Name: RFC
    6. Source: None
4. **Preprocessing job**
    7. **Type: ** pre-processing
    8. **Docker-image: **docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14
    9. **Script: **python featureset.py --fs titanic-fs
        3. titanic is featureset name, the user can change the name. 
    10. **Code: **titanic
    11. **Input dataset: **titanic
        4. **Mount-point: **/opt/dkube/input
    12. **Output featureset: **titanic-fs
        5. **Mount-Point: **/opt/dkube/output
5. **Training job**
    13. **Type: ** training
    14. **Framework: **Tensorflow
    15. **Framework Version: **1.14
    16. **Image** : docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14
    17. **Script: **python model.py
    18. **Project: **titanic
    19. **Input featureset: **titanic-fs
        6. **Mount-point: **/opt/dkube/input
    20. **Output model: **RFC
        7. **Mount-Point: **/opt/dkube/output
6. **Pipeline run: **
    21. Download pipeline: [https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/sklearn/titanic/pipeline/dkube_titanic_pl.tar.gz](https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/sklearn/titanic/pipeline/dkube_titanic_pl.tar.gz) 
    22. Upload and create a run, 
        8. Fill auth token from developer settings
        9. **training_program: **titanic (This is the project name and should exist)
        10. **preprocessing_dataset: **titanic (This is the dataset name and should exist)
        11. **training_featureset: **titanic-fs  (This is the feature set name and should exist)
        12. **training_output_model: **RFC (This is the model name and should exist)


## **Test Inference:**



1. Go to the model and click test inference,
2. Check transformer option, and replace the transformer script with **sklearn/titanic/program/transformer.py**
3. Choose CPU, and submit.
4. Go to https://&lt;URL>:32222/inference
    1. Copy the model serving URL from the test inference tab.
    2. Copy the auth token from developer settings, 
    3. Select model type sk-stock
    4. Copy the contents of [https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/sklearn/titanic/data/titanic_sample.csv](https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/sklearn/titanic/data/titanic_sample.csv) and save then as CSV, and upload.
    5. Click predict.
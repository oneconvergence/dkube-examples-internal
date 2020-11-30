## **MNIST feature-set example**



1. **Add Code**
    1. Name: mnist-fs
    2. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/program](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/program) 
2. **Add dataset/featureset:**
    3. Dataset Name: mnist
        1. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/data](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/data) 
    4. Featureset Name: mnist-fs
        2. Featurespec upload: none
3. **Create Model:**
    5. Name: mnist
    6. Source: None
4. **Preprocessing job**
    7. **Type: ** pre-procesing
    8. **Docker-image: **docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14
    9. **Script: **python featureset.py
    10. **Code: **mnist-fs
    11. **Input dataset: **mnist
        3. **Mount-point: **/opt/dkube/input
    12. **Output featureset: **mnist-fs
        4. **Mount-Point: **/opt/dkube/output
5. **Training job**
    13. **Type: ** training
    14. **Framework: **Tensorflow
    15. **Framework Version: **1.14
    16. **Image** : docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14
    17. **Script: **python model.py
    18. **Project: **mnist-fs
    19. **Input featureset: **mnist-fs
        5. **Mount-point: **/opt/dkube/input
    20. **Output model: **mnist
        6. **Mount-Point: **/opt/dkube/output
6. **Pipeline run: **
    21. Download pipeline: [https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/tf/classification/mnist-fs/digits/pipeline/dkube_mnist_fs_pl.tar.gz](https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/tf/classification/mnist-fs/digits/pipeline/dkube_mnist_fs_pl.tar.gz) 
    22. Upload and create a run, 
        7. Fill auth token from developer settings
        8. **training_program: **mnist-fs (This is the project name and should exist)
        9. **preprocessing_dataset: **mnist (This is the dataset name and should exist)
        10. **training_featureset: **mnist-fs  (This is the feature set name and should exist)
        11. **training_output_model: **mnist (This is the model name and should exist)


## **Test Inference:**



1. Go to the model and click test inference,
2. Check transformer option, and replace the transformer script with **tf/classification/mnist-fs/digits/transformer/transformer.py**
3. Choose CPU, and submit.
4. Go to https://&lt;URL>:32222/inference
    1. Copy the model serving URL from the test inference tab.
    2. Copy the auth token from developer settings, 
    3. Download any .png image from, [https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/inference](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/inference) and upload.
    4. Click predict.
## **MNIST feature-set example**



1. **Add Code** \
&nbsp;&nbsp; Name: mnist \
&nbsp;&nbsp; Git URL: [https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist-fs/digits/classifier/program](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/program) 
2. **Add dataset/featureset:** \
&nbsp;&nbsp;2.1 Dataset Name: mnist \
&nbsp;&nbsp;&nbsp;&nbsp; Git URL: [https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist-fs/digits/classifier/data](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/data) \
&nbsp;&nbsp;2.2 Featureset Name: mnist-fs \
&nbsp;&nbsp;&nbsp;&nbsp; Featurespec upload: none 
3. **Create Model:** \
&nbsp;&nbsp; Name: mnist \
&nbsp;&nbsp; Source: None 
4. **Preprocessing job** \
&nbsp;&nbsp; **Type:** pre-procesing \
&nbsp;&nbsp; **Docker-image:** docker.io/ocdr/d3-datascience-tf-cpu:v1.14 \
&nbsp;&nbsp; **Script:** python featureset.py --fs mnist-fs \
&nbsp;&nbsp; **Code:** mnist \
&nbsp;&nbsp; **Input dataset:** mnist \
&nbsp;&nbsp;&nbsp;&nbsp; **Mount-point:** /opt/dkube/input \
&nbsp;&nbsp; **Output featureset:** mnist-fs \
&nbsp;&nbsp;&nbsp;&nbsp; **Mount-Point:** /opt/dkube/output 
5. **Training job** \
&nbsp;&nbsp; **Type:** training \
&nbsp;&nbsp; **Framework:** Tensorflow \
&nbsp;&nbsp; **Framework Version:** 1.14 \
&nbsp;&nbsp; **Script:** python model.py --fs mnist-fs \
&nbsp;&nbsp; **Code:** mnist \
&nbsp;&nbsp; **Input featureset:** mnist-fs \
&nbsp;&nbsp;&nbsp;&nbsp; **Mount-point:** /opt/dkube/input \
&nbsp;&nbsp; **Output model:** mnist \
&nbsp;&nbsp;&nbsp;&nbsp; **Mount-Point:** /opt/dkube/output 
6. **Pipeline run:** \
&nbsp;&nbsp; Download [https://raw.githubusercontent.com/oneconvergence/dkube-examples-internal/master/tf/classification/mnist-fs/digits/pipeline/dkube-mnist-fs.ipynb] into DKube Notebook IDE and run all the cells. This auto fills values for pipeline parameters as follows and creates a run under "Dkube - Mnist Featureset" experiment. \
&nbsp;&nbsp;&nbsp;&nbsp; Fills in auth token from DKUBE_USER_ACCESS_TOKEN environment variable in Notebook  \
&nbsp;&nbsp;&nbsp;&nbsp; **training_program:** mnist (This is the code name and should exist) \
&nbsp;&nbsp;&nbsp;&nbsp; **preprocessing_dataset:** mnist (This is the dataset name and should exist) \
&nbsp;&nbsp;&nbsp;&nbsp; **training_featureset:** mnist-fs  (This is the feature set name and should exist) \
&nbsp;&nbsp;&nbsp;&nbsp; **training_output_model:** mnist (This is the model name and should exist) 


## **Test Inference:**



1. Go to the model and click test inference, 
2. Check transformer option, and replace the transformer script with **tf/classification/mnist-fs/digits/transformer/transformer.py** 
3. Choose CPU, and submit. 
4. Go to https://&lt;URL>:32222/inference \
&nbsp;&nbsp;1. Copy the model serving URL from the test inference tab. \
&nbsp;&nbsp;2. Copy the auth token from developer settings, \
&nbsp;&nbsp;3. Download any .png image from, [https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist-fs/digits/inference](https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/classification/mnist-fs/digits/inference) and upload. \
&nbsp;&nbsp;4. Click predict.

## **MNIST feature-set example**



1. **Add Code**
&nbsp;&nbsp;1. Name: mnist-fs \
&nbsp;&nbsp;2. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/program](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/program) \
2. **Add dataset/featureset:**
&nbsp;&nbsp;3. Dataset Name: mnist \
&nbsp;&nbsp;&nbsp;&nbsp;1. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/data](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/classifier/data) \
&nbsp;&nbsp;4. Featureset Name: mnist-fs \
&nbsp;&nbsp;&nbsp;&nbsp;2. Featurespec upload: none \
3. **Create Model:**
&nbsp;&nbsp;5. Name: mnist \
&nbsp;&nbsp;6. Source: None \
4. **Preprocessing job**
&nbsp;&nbsp;7. **Type:** pre-procesing \
&nbsp;&nbsp;8. **Docker-image:** docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14 \
&nbsp;&nbsp;9. **Script:** python featureset.py \
&nbsp;&nbsp;10. **Code:** mnist-fs \
&nbsp;&nbsp;11. **Input dataset:** mnist \
&nbsp;&nbsp;&nbsp;&nbsp;3. **Mount-point:** /opt/dkube/input \
&nbsp;&nbsp;12. **Output featureset:** mnist-fs \
&nbsp;&nbsp;&nbsp;&nbsp;4. **Mount-Point:** /opt/dkube/output \
5. **Training job**
&nbsp;&nbsp;13. **Type:** training \
&nbsp;&nbsp;14. **Framework:** Tensorflow \
&nbsp;&nbsp;15. **Framework Version:** 1.14 \
&nbsp;&nbsp;16. **Image** : docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14 \
&nbsp;&nbsp;17. **Script:** python model.py \
&nbsp;&nbsp;18. **Project:** mnist-fs \
&nbsp;&nbsp;19. **Input featureset:** mnist-fs \
&nbsp;&nbsp;&nbsp;&nbsp;5. **Mount-point:** /opt/dkube/input \
&nbsp;&nbsp;20. **Output model:** mnist \
&nbsp;&nbsp;&nbsp;&nbsp;6. **Mount-Point:** /opt/dkube/output \
6. **Pipeline run:**
&nbsp;&nbsp;21. Download pipeline: [https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/tf/classification/mnist-fs/digits/pipeline/dkube_mnist_fs_pl.tar.gz](https://raw.githubusercontent.com/oneconvergence/dkube-examples/master/tf/classification/mnist-fs/digits/pipeline/dkube_mnist_fs_pl.tar.gz) \
&nbsp;&nbsp;22. Upload and create a run, \
&nbsp;&nbsp;&nbsp;&nbsp;7. Fill auth token from developer settings \
&nbsp;&nbsp;&nbsp;&nbsp;8. **training_program:** mnist-fs (This is the project name and should exist) \
&nbsp;&nbsp;&nbsp;&nbsp;9. **preprocessing_dataset:** mnist (This is the dataset name and should exist) \
&nbsp;&nbsp;&nbsp;&nbsp;10. **training_featureset:** mnist-fs  (This is the feature set name and should exist) \
&nbsp;&nbsp;&nbsp;&nbsp;11. **training_output_model:** mnist (This is the model name and should exist) \


## **Test Inference:**



1. Go to the model and click test inference,
2. Check transformer option, and replace the transformer script with **tf/classification/mnist-fs/digits/transformer/transformer.py**
3. Choose CPU, and submit.
4. Go to https://&lt;URL>:32222/inference
&nbsp;&nbsp;1. Copy the model serving URL from the test inference tab.
&nbsp;&nbsp;2. Copy the auth token from developer settings, 
&nbsp;&nbsp;3. Download any .png image from, [https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/inference](https://github.com/oneconvergence/dkube-examples/tree/master/tf/classification/mnist-fs/digits/inference) and upload.
&nbsp;&nbsp;4. Click predict.
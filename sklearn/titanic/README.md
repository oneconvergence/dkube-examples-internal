## **Titanic feature-set example**



1. **Add Code** \
&nbsp;1. Name: titanic \
&nbsp;2. Git URL: [https://github.com/oneconvergence/dkube-examples/tree/titanic_demo/sklearn/titanic/program](https://github.com/oneconvergence/dkube-examples/tree/titanic_demo/sklearn/titanic/program) 
2. **Add dataset and featureset:** \
&nbsp;1. Dataset Name: titanic \
&nbsp;&nbsp;1.1 Git URL: [https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/data](https://github.com/oneconvergence/dkube-examples/tree/master/sklearn/titanic/data) \
&nbsp;2. Featureset Name: titanic-train-fs \
&nbsp;&nbsp;2.2 Featurespec upload: none \
&nbsp;2. Featureset Name: titanic-test-fs \
&nbsp;&nbsp;2.2 Featurespec upload: none 
3. **Create Model:** \
&nbsp;1. Name: titanic \
&nbsp;2. Source: None
4. **Preprocessing job** \
&nbsp;1. **Type:** pre-processing \
&nbsp;2. **Docker-image:** docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14 \
&nbsp;3. **Script:** python preprocessing.py --train_fs titanic-train-fs --test_fs titanic-test-fs \
&nbsp;&nbsp;3.1. titanic is featureset name, the user can change the name. \
&nbsp;4. **Code:** titanic \
&nbsp;5. **Input dataset:** titanic \
&nbsp;&nbsp;5.1. **Mount-point:** /opt/dkube/input \
&nbsp;6. **Output featureset:** titanic-train-fs, titanic-test-fs \
&nbsp;&nbsp;6.1. **Mount-Point:** /opt/dkube/output/train, /opt/dkube/output/test
5. **Training job** \
&nbsp;1. **Type:** training \
&nbsp;2. **Framework:** sklearn \
&nbsp;3. **Framework Version:** 0.23.2 \
&nbsp;4. **Image** : docker.io/ocdr/d3-datascience-tf-cpu:fs-v1.14 \
&nbsp;5. **Script:** python training.py \
&nbsp;6. **Project:** titanic \
&nbsp;7. **Input featureset:** titanic-train-fs, titanic-test-fs\
&nbsp;&nbsp;7.1. **Respective Mount-point:** /opt/dkube/input, /opt/dkube/test \
&nbsp;8. **Output model:** titanic \
&nbsp;&nbsp;8.1. **Mount-Point:** /opt/dkube/output
## **Steps for running the Mnist Example in R-Studio**

Skip steps 1 and 2, if you have already created a Project and Model based on *[https://github.com/oneconvergence/dkube-examples/blob/master/R/classification/mnist/README.md](https://github.com/oneconvergence/dkube-examples/blob/master/R/classification/mnist/README.md)*

## Step 1: Create a Code
1. Click *Repos* side menu option.
2. Click *+Code* button.
3. Select source as *Git*.
4. Enter a unique name say *r-examples*
5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/R/classification
 ](https://github.com/oneconvergence/dkube-examples/tree/master/R/classification)* in the URL text box.
6. Branch: master
7. Click *Add Code* button.
8. Code will be created and imported in Dkube. Progress of import can be seen.
9. Please wait till status turns to *ready*

## Step 2: Create a model
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Enter a unique name say *r-mnist*.
 4. Select Versioning as DVS. 
 5. Select Model store as default.
 6. Select Model Source as None.
 7. Click the Add Model button.
 8. Model will be created on Dkube.
 9. Please wait till status turns to ready.

## Step 3: Launch R-studio and Run
 1. Launch an R-studio with project r-examples and input model r-mnist (/opt/dkube/model) and run the programs in the r-studio. 
 2. Open file workspace/<project-name>/R/classification/rstudio/mnist.R
 3. Run the code from Code menu -> Run region -> Run All


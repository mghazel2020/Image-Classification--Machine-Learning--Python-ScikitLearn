# Image-Classification--Machine-Learning--ScikitLearn

# 1. Objectives

The objective of this project is to demonstrate how to use scikit-learn to recognize images of hand-written digits, using various Machine Learning (ML) built-in image classification functionalities and compare their performance. We shall apply a standard ML via Scikit-Learn process, deomstrate the implementation and output of each step. 

# 2. Data Set

Scikit-learn comes with a few small datasets that do not require to download any file from some external website. We shall use the the DIGITS dataset, which allows us to quickly train and deeploy various ML classification algorithms implemented in the scikit and assess and compare their performance. Unlike the MINIST data set, the DIGIT data set is however too small to be representative of real world machine learning image classification tasks. After demonstrating the basics of logisitic regression, one can easily the MNIST Handwritten digit database. 

A quick summary of the DIGITS data set is as follows:

*   The DIGITS dataset of handwritten digits (0-9) in the Scikit-Learn database
*   It contains 1,797 labeled digits examples, with about 180 examples per digit.
*   It is a small subset of a larger set available from MNIST, which contains 70,000 images. 
*   The DIGITS have been size-normalized and centered in a fixed-size grayscale image (8x8x1 pixels)
*   This is different from the full MNIST data set, where the grayscale images are 28x28x1 pixels.

Next, we oultline the standard standard ML via Scikit-Learn process, which is adopted as our appraoch.
    
# 4. Approach

1. Load the input data set and split into training and test data subsets 

2. Select and build a ML classification algorithm from scikit-learn API:

  * We shall explore the performance of the following ML algorithms:

    * Support Vector Machines (SVM)
    * LogisticRegression
    * DecisionTree and Random Forest
    * Multilayer Perceptron (MLP)
    * Stochasti Gradeint Descent (SGD)

3. Train the selected ML model

3. Deploy the trained on the test data

4. Evaluate the performance of the trained model using evaluation metrics:

* Accuracy
* Confusion Matrix
* Other metrics derived form the confusion matrix

5. Perform hyperparameter search and fine-tuning to identify more optimal ML model paramaters:

* Explore grid-search and random-search to identify the parameter
* Train the selected model using the identified hyperparameters
* Deploy the improved trained model on the test data
* Evaluate the performance of the improved model. 

We shall demonstrate the above process for the SVM model and then tabulate and compare the performance of the 5 different implemented ML classification algorithms. 

# 3. Computing Framework

The 5 notebooks in **./code/** are Google Colaboratory (Colab) notebooks. Colab is a Python development environment, with pre-installed software requirements and dependencies,  that runs in the browser using Google Cloud ([Google Colaboratoro](https://colab.research.google.com)). We easily open and execute each notebook using Google Colab, as follows:

* Download the notebooks and copy to them to your Google Drive, which is freely available if you have a GMAIL account.
* Once copied to your Google Drive, you can easily open a notebook as follows:
  * Right-click on the notebook
  * Select: Open With --> Google Colaboratory
* Once you open a notebook via Google Colaboratory you can then easily explore, edit, execute each cell and code blocks.

# 3. Scikit-Learn ML Process Demonstration for SVM

# 5. Comparison of the 5 ML classification Algorithms

# 6. Conclusions

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

A sample of the DIGITS images are illustrated next.

![image](https://user-images.githubusercontent.com/80174045/111051767-6bea7900-840a-11eb-99de-ecf39a3c76aa.png)

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

The 5 notebooks in **./code/** are Google Colaboratory (Colab) notebooks. Colab is a Python development environment, with pre-installed software requirements and dependencies,  that runs in the browser using Google Cloud ([Google Colaboratory](https://colab.research.google.com)). We easily open and execute each notebook using Google Colab, as follows:

* Download the notebooks and copy to them to your Google Drive, which is freely available if you have a GMAIL account.
* Once copied to your Google Drive, you can easily open a notebook as follows:
  * Right-click on the notebook
  * Select: Open With --> Google Colaboratory
* Once you open a notebook via Google Colaboratory you can then easily explore, edit, execute each cell and code blocks.

# 4. Scikit-Learn ML Process Demonstration for SVM

## 4.1. Load and explore the input DIGITS data set and split into training and test data subsets 

First, load the DIGITS data set and examine the numbe rof examples and their classification.

```python
# Download the digits dataset from sklearn datasets
digits = datasets.load_digits()
# set the feature vector X
X = digits.data
# Display the feature vector shape
print('The shape of the features: X = ({0}, {1})'.format(X.shape[0], X.shape[1]))
# the total number of images
total_num_images = X.shape[0]
print("MNIST contains {} images.".format(total_num_images))
```
[Output]  
```
The shape of the targets: y = 1797
The unique targets: [0 1 2 3 4 5 6 7 8 9]
```
We then visualize randomly selected 25 images and their associated labels. 

```python
"""
# A utility function to visualize multiple images:
"""
def visualize_images_and_labels(num_visualized_images = 25):
  """To visualize images.

      Keyword arguments:
         - num_visualized_images -- the number of visualized images (deafult 20)
      Return:
         - None
  """
  # the suplot grid shape
  num_rows = 5
  # the number of columns
  num_cols = num_visualized_images // num_rows
  # setup the subplots axes
  fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4, 5))
  # set a seed random number generator for reproducible results
  seed(random_state_seed)
  # iterate over the sub-plots
  for row in range(num_rows):
      for col in range(num_cols):
        # get the next figure axis
        ax = axes[row, col];
        # turn-off subplot axis
        ax.set_axis_off()
        # generate a random image counter
        counter = randint(0, total_num_images)
        # get the flattened image and reshape it into 8x8 pixls
        image = X[counter,:].reshape(8,8)
        # get the target associated with the image
        label = y[counter]
        # display the image
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        # set the title showing the image label
        ax.set_title('y =' + str(label), size = 8)
```

```python
# call the function to visualize the input images
visualize_images_and_labels(num_visualized_images)
```
[Output]  

![image](https://user-images.githubusercontent.com/80174045/111051839-dac7d200-840a-11eb-9795-6f69400af64a.png)

We  Split the data into training and testing subsets:
*   Use 20 percent of data set for testing  
*   Use 80 percent of data set for training.

```python
# fraction of the test data
test_data_fraction = 0.20
# Split the data into training (0.2) and testing (0.2) data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_fraction, random_state = random_state_seed, shuffle=False)
```
```python
# check the number of training images
num_train_images = X_train.shape[0]
print('Number of train images = {}'.format(num_train_images))
# check the number of test images
num_test_images = X_test.shape[0]
print('Number of test images = {}'.format(num_test_images))
```
[Output] 
```
Number of train images = 1347
Number of test images = 450
```
We also examine the number of examples of each class in the training and testing subsets in order to make sure that the 10 classes are approximately balanced. That is, we typically want to make sure that we have similar number of examples of each digit in the training data subset because most ofthe ML classification algorithms used in this work expect training data set with balanced classes. 

[Output]  

![image](https://user-images.githubusercontent.com/80174045/111039808-e0122600-83e4-11eb-9506-f2c912394ce6.png)

We now normaize the training and testing images:
* Grayscale image values range from 0 to 255
* Scale by 255 to normalize the range betwen 0 and 1
* We experimented with normalizing the input images
* We set a flag to indicate whether or not we normaized the input images to see if normalizing the input training and test images improves the model performance.
```python
# nomalize the train and test images if desired
if ( normalize_images_flag == 1):
  # normalize X_train
  X_train =  X_train / 255
  # normalize X_test
  X_test =  X_test / 255
```

## 4.2. Import and instantiate the Scikit-Learn ML model:

We import and then instantiate the SVM model:

```python
# Instantiate the support vector (SVM) classifier
svm_model = svm.SVC()
```

## 4.3. Train the selected SVM model using the training data:

Training a SVM model on the training images can be done as follows:

```python
# Instantiate the support vector (SVM) classifier
svm_model = svm.SVC()
# Train the SVM model and printout its configuration parameters
svm_model.fit(X_train, y_train)
```
[Output]  
```
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

## 4.4 Step 4: Deploy the trained SVM model to predict the classification of the test data:

Deploying the trained SVM to classify the test images is straightforward, as follows:

```python
# Instantiate the support vector (SVM) classifier
svm_model = svm.SVC()
# Train the SVM model and printout its configuration parameters
svm_model.fit(X_train, y_train)
```
[Output]  
```
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```
We now visualize 25 randomly selected test images and their associated predicted labels.

![image](https://user-images.githubusercontent.com/80174045/111051911-7eb17d80-840b-11eb-90dd-cc5dff68bb7c.png)


## 4.5 Step 5: Evaluate the performance of the trained model:

Evaluate the performance of the trained model using various performance metrics:
* The model accuracy score
* The classification report summary
* The confusion matrix

### 4.5.1 Accuracy

The model accuracy captures how the model performs on new data (test set) in one value, in terms of the fraction of correct predictions:

```math
Accuracy = \frac{\mbox{correct predictions}}{\mbox{total number of test images}}
```

```python
# Overall accuracy:
# - accuracy =  fraction of correct predictions =  correct predictions / total number of data points 
# - Basically, how the model performs on new data (test set)
# Use score method to get accuracy of model
score = svm_model.score(X_test, y_test)
print('The overall accuracy = ' + str(score))
```
[Output]  
```
The overall accuracy = 0.9416666666666667
```

### 4.5.2 Classification report summary:

The classification_report builds a text report showing the main classification metrics:

```python
# Generate a classification_report
print(f"Classification report for SVM classifier {svm_model}:\n"
      f"{metrics.classification_report(y_test, svm_yhat)}\n")
```
[Output]  
```
Classification report for SVM classifier SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False):
              precision    recall  f1-score   support

           0       1.00      0.97      0.99        35
           1       0.97      1.00      0.99        36
           2       1.00      1.00      1.00        35
           3       0.96      0.73      0.83        37
           4       0.97      0.92      0.94        37
           5       0.93      1.00      0.96        37
           6       1.00      1.00      1.00        37
           7       0.92      0.97      0.95        36
           8       0.78      0.94      0.85        33
           9       0.92      0.89      0.90        37

    accuracy                           0.94       360
   macro avg       0.94      0.94      0.94       360
weighted avg       0.95      0.94      0.94       360
```

### 4.5.3 Confusion matrix:

The confusion matrix plots the true digit values and their associated predicted digit labels:

```python
# We can also plot a confusion matrix of the true digit values and the predicted digit values.
disp = metrics.plot_confusion_matrix(svm_model, X_test, y_test)
# display the confusion matrix
print(f"Confusion matrix:\n{disp.confusion_matrix}")
# visualize the confusion matrix
disp.figure_.suptitle("Confusion Matrix")
```
[Output]  
```
![image](https://user-images.githubusercontent.com/80174045/111040552-3af94c80-83e8-11eb-97dd-807f5ab1ad56.png)

We should also examine some of the misclassified digits, in order to gain some insights of the reasons the trained SVM model has misclassified them.

```python
# We can also plot a confusion matrix of the true digit values and the predicted digit values.
disp = metrics.plot_confusion_matrix(svm_model, X_test, y_test)
# display the confusion matrix
print(f"Confusion matrix:\n{disp.confusion_matrix}")
# visualize the confusion matrix
disp.figure_.suptitle("Confusion Matrix")
```
[Output] 

![image](https://user-images.githubusercontent.com/80174045/111040630-b529d100-83e8-11eb-9e26-e5ce47fd10ea.png)

## 4.6 Step 6: Perform hyperparameter search and fine-tuning to identify more optimal ML model paramaters and improve the model performance:

The SVM image classification model was training using default hyper-parameters:

* These configuration paramaters may not be optimal for our DIGITS data set
* Thus, there is a need to expriment with different parameter values to see if we can achieve a better model performane:
* We shall now explore grid-search and random-search to identify the parameter
  * Train the selected model using the identified hyperparameters
  * Deploy the improved trained on the test data
  * Evaluate the performance of the improved model.

The configuration hyperparamaters of the svm.SVC() model as as follows:

[Output]  
```
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

We explore varying the values of the following 2 paramaters:

* C:

  * Controls the cost of mis-classification onthe training data
  * Larger C values yield low bias and high variance
  * Smaller C values yield higher bias and lower variance

* gamma:

  * Controls the shape of the RBF kernel (default kernel)
   * Small gamma, Gaussian with large variance and lower bias
  * Larger gamma value lead to high bias and lower variance.

4.6.1 Use Grid-Search to perform hyper-paramater fine-tuning:

```python
# define the parameters search grid
tuned_parameters = [{
            'kernel': ['rbf'], 
            'gamma':  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3],
            'C':      [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]
        }]
# Instantiate the SVM model with grid-search
svm_model_grid = GridSearchCV(svm.SVC(), tuned_parameters[0], verbose=3)
# fit the model 
svm_model_grid.fit(X_train, y_train)
```

After the grid-search as completed, we obtain the best hyper-paramaters and the improved SVM performance as follows:

```python
# get the best parameters combination
svm_model_grid.best_params_
```

[Output]  
```
{'C': 10.0, 'gamma': 0.001, 'kernel': 'rbf'}
```


```python
# get the best estimated SVM model parameters
svm_model_grid.best_estimator_
```

[Output]  
```
SVC(C=10.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

```python
# get the best score (accuracy)
svm_model_grid.best_score_
```

[Output]  
```
0.9770349399922571
```

4.6.2 Use Random-Search to perform hyper-paramater fine-tuning:

```python
# define the parameters search grid
tuned_parameters = [{
            'kernel': ['rbf'], 
            'gamma':  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3],
            'C':      [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]
        }]
# Instantiate the SVM model with randomized-search
svm_model_random = RandomizedSearchCV(svm.SVC(), tuned_parameters[0], verbose=3, n_iter = 10000, cv = 2, random_state=101 , n_jobs = -1)
# fit the model 
svm_model_random.fit(X_train, y_train)
```

After the grid-search as completed, we obtain the best hyper-paramaters and the improved SVM performance as follows:

```python
# get the best parameters combination
svm_model_grid.best_params_
```

[Output]  
```
{'C': 10.0, 'gamma': 0.001, 'kernel': 'rbf'}
```


```python
# get the best estimated SVM model parameters
svm_model_grid.best_estimator_
```

[Output]  
```
SVC(C=10.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

```python
# get the best score (accuracy)
svm_model_grid.best_score_
```

[Output]  
```
0.9638212311280369
```

4.6.3 Final Assessment

In this final assessment, we comare the performance if teh trained SVM model using default paramaters, as well as the more optimal parameters, as identified by the Grid-Search and the Random-Serach algorithms. Clearly, apply the serach algorithms has resulted in using more suitable hyperperameters for our DIGITS data set and yielding better classification accuracy.


| model_name       | Default Paramaters | Grid-Search Parameters | Random-Search Parameters
|------------------|-------------------|--------------------|---------------------------------|
|Accuracy          | 0.9416666666666667       | 0.9770349399922571               | 0.9638212311280369 |



# 5. Comparison of the 5 ML classification Algorithms

# 6. Conclusions

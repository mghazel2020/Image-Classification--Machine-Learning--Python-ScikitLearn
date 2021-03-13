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
        ax.set_title('y =' + str(label))
```

```python
# call the function to visualize the input images
visualize_images_and_labels(num_visualized_images)
```
[Output]  

![image](https://user-images.githubusercontent.com/80174045/111039555-9248ee00-83e3-11eb-9514-c4da3a6fc2c2.png)

We  Split the data into training and testing subsets:
*   Use 20 percent of data set for testing  
*   Use 80 percent of data set for training.

```python
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
We also examine the number of examples of each class in the training uset in order to make sure that the 10 classes are approximately balanced. That is, we typically want to make sure that we have similar number of examples of each digit in the training data subset because most ofthe ML classification algorithms used in this work expect training data set with balanced classes. 

[Output]  

![image](https://user-images.githubusercontent.com/80174045/111039808-e0122600-83e4-11eb-9506-f2c912394ce6.png)

We now normaize the training and testing images:
* Grayscale image values range from 0 to 255
* Scale by 255 to normalize the range betwen 0 and 1
* We experimented with normalizing the input images
  * Therefore, we set a flag to indicate whether or not we normaized the input images.
```python
# nomalize the train and test images if desired
if ( normalize_images_flag == 1):
  # normalize X_train
  X_train =  X_train / 255
  # normalize X_test
  X_test =  X_test / 255
```

# 5. Comparison of the 5 ML classification Algorithms

# 6. Conclusions

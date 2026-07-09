# Project Brief


PROJECT 2
 (cid:1082)(cid:1120)(cid:1204)
Indian Institute of Technology Hyderabad
This repository contains the code, data, figures, and derivations for the complete analysis workflow. Keep the
codes (.py or .ipynb) and plots (.png, .jpg, .jpeg, .pdf) of all the modules into one single folder, compress this folder, and
1 Training an Artificial Neural Network
In this task, you will learn how to train an artificial neural network (or simply neural network), a key ingredient in the
implementation of many deep learning tasks in modern machine learning applications.
1.1 Dataset Preparation
We shall first prepare our dataset.
1. Using pandas, load the given dataset data_Q1.csv.
2. Using the .to_numpy() method, convert each column of the dataframe to a numpy array and store them in arrays
named training_x1, training_x2, and training_y respectively. Let N denote the length of each array.
3. Create a numpy array named training_x using the following code snippet:
training_x = np.array([[x1,x2,1] for x1,x2 in zip(training_x1,training_x2)])
1.2 Neural Network Preliminaries
Our neural network will consist of one input layer with 2 nodes (excluding bias), one hidden layer with 3 nodes (excluding
bias), and an output layer. Below is the model for our neural network.
X W(1) H  Z W(2) O  yb.
Here:
 X is the input data (coming from training_data) of size N  3.
 W (1) = [W (1) : k, l  {1, 2, 3}] is a weight matrix of size 3  3, connecting the input layer and hidden layer.
k,l
 W (2) = [W (2) : k  {1, 2, 3, 4}] is a weight matrix of size 4  1 between the hidden layer and the output layer.
k,1
  is the sigmoid activation function, defined by (x) = 1 .
1+ex
 H = X  W (1) is a matrix of size N  3 obtained at the hidden layer, before the activation function  is applied.
 Z is a matrix of size N  4 obtained at the hidden layer after the activation function is applied to each entry of H,
followed by appending the bias column at the end.
 O = Z  W (2) is a matrix of size N  1 obtained at the output layer before the activation function is applied.
 yb = (O) is a matrix of size N  1 containing the predicted values.


To measure the performance of our neural network, we will use the socalled meansquarederror loss, defined as
XN
1
L(Y, yb) := (Y  yb )2,
2N i i
i=1
where N is the number of data points, Y represents the true value for the ith data point (coming from the array training_y),
i
and yb represents the predicted value for the ith data point.
i
1.3 Definitions and Initialisations
1. Add the following line to your code:
np.random.seed(123)
This will lead to reproducible variable values when working with random data.
2. Define a function called sigmoid which takes one single input x and returns 1 . You will be calling this function
1+ex
on each entry of a matrix.
3. Define a function called grad_sigmoid which takes a single input x and returns the derivative of (x). Again, you
will be calling this function on each entry of a matrix.
4. Initialisation of weights matrices W (1) and W (2):
(a) Given a random floating point number   [0, 1], we can generate a random floating point in [a, b] as follows:
a + (b  a).
(b) The weight matrix W (1) is represented by weights1. Initialize this 3  3 matrix with random floating point
numbers lying between [1, 1].
(c) The weight matrix W (2) is represented by weights2. Initialize this 4  1 matrix with random floating point
numbers lying between [1, 1].
5. The learning rate, say , is a hyperparameter which controls the size of the steps you take while descending along
the gradient. Initialize the learning rate to 0.05.
6. An epoch is one complete pass of the data through the network. Initialize max_epochs to 100.
7. Create an empty array to store the training errors after each epoch. Name the array training_error.
1.4 Training the Neural Network
Repeat the following over max_epochs many epochs.
1. Forward pass.
(a) Initialise X to training_x.
(b) Set H as the matrix multiplication of X and W (1).
(c) Set Z as (H), where (H) denotes the matrix obtained by applying the sigmoid function on each element of
H.
(d) Append an allones column to Z. Store the result in Z again.
(e) Set O as the product of Z and W (2).
(f) Set yb as (O).
(g) Compute the loss between the true values in training_y and the predicted values in yb. Append the loss to the
array training_error.
2. Back propagation.
(a) Calculating the gradients for weights2.


i. Calculate the partial derivative of the error function with respect to W (2) as follows:
k,1
L XN L yb O
=  i  i , k  {1, 2, 3, 4}. (1)
W (2) yb i O i W (2)
k,1 i=1 k,1
Attach your calculations (in a .pdf file).
ii. Implement the gradient you calculated and store it in a matrix gradient2 which will have the same shape
as weights2.
(b) Calculating the gradients for weights1:
i. Calculate the partial derivative of the loss function with respect to W (1) as follows:
k,l
L XN L yb O Z H
=  i  i  i,l  i,l , k, l  {1, 2, 3}. (2)
W (1) yb i O i Z i,l H i,l W (1)
k,l i=1 k,l
Attach your calculations (in a .pdf file).
You may use the below snippet of code for calculating the gradients in (2):
temp = -(training_y - y_hat) * grad_sigmoid(O) * grad_sigmoid(H)
temp = temp @ np.diag(weights2[:-1].reshape(-1))
gradient2 = training_x.T @ temp
gradient2 /= len(training_y)
Following the above snippet, generate your own snippet of code for implementing the gradients in (1).
ii. Implement the gradient you calculated and store it in a matrix gradient1 which will have the same shape
as weights1.
(c) Gradient descent:
i. Update each entry of weights2 matrix (i.e., the weights matrix W (2)) using the rule
L
W (2)  W (2)   , k  {1, 2, 3, 4}.
k,1 k,1 W (2)
k,1
ii. Update each entry of weights1 matrix (i.e., the weights matrix W (1)) using the rule
L
W (1)  W (1)   , k, l  {1, 2, 3}.
k,l k,l W (1)
k,l
In both of the above equations,  denotes the learning rate parameter.
1.5 Plotting the Training Loss
Plot the training loss on the yaxis and the epochs on the xaxis. Label the axes and provide a title to the plot.


2 Classification using the kNearest Neighbors Technique
In this task, you will use the popular knearest neighbors method to learn the classes in a labeled dataset and predict the
class of a new data point. Classification is an ubiquitous supervised learning technique that finds use in several daytoday
machine learning applications.
2.1 The Abalone Dataset
We will work with the abalone dataset. Add the following snippet of code to load the data into a pandas dataframe.
import pandas as pd
url = ``https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data''
abalone = pd.read_csv(url, header=None)
The age of an abalone can be found by cutting its shell and counting the number of rings on the shell. In the abalone
dataset, you can find the age measurements of a large number of abalones along with a lot of other physical measurements.
The goal of this task is to learn a model that can predict the age of an abalone as a function of its physical measurements.
2.2 Data PreProcessing and Cleaning
Notice that we have removed the column names when importing the data from the above url. Append the column names
in the below array to the abalone dataframe created earlier.
(cid:2)
column_names = Sex, Length, Diameter, Height, Whole weight,
(cid:3)
Shucked weight, Viscera weight, Shell weight, Rings
One of the columns in the dataset is Sex of the abalone, which does not directly influence its age. Remove this column
from the dataset.
2.3 Training and Test Data
Now, we shall use Pythons scikit-learn to split our dataset into two parts. One part, called the training dataset, will be
used to learn the classes in the data, and the other part, called the test dataset, will be used to evaluate the performance
of the learning task.
Use the below snippet of code to create training and test datasets.
from sklearn.model_selection import train_test_split
(cid:0)
X_train, X_test, y_train, y_test = train_test_split
(cid:1)
X, y, test_size=0.3, random_state=333
In the above snippet, you should substitute y with the Rings column and X with the remaining columns in abalone. Use
the .values() method to remove the column names before inputting the data values to X and y in the above snippet.
Note: The above snippet creates a test dataset with 30% of the total data points and a training dataset with 70% of the total
data points.
2.4 Implementing the kNearest Neighbors Algorithm
Consider a new abalone with physical measurements as given in Table 1. Our task is to predict its age using the knearest
neighbors technique.
1. Using NumPys linalg.norm() method, compute the Euclidean distance between the each row of X_train and
that of the new data point (viewed as a row with same column names as X_train). Let the distances be stored in a
variable called distances.
2. Sort the values in distances in increasing order of their values, and get the indices of the smallest 3 values using
the .argsort() method. These indices correspond to the indices of the 3 nearest neighbors in the abalone dataset
to the new data point.


Variable Value
Length 0.569552
Diameter 0.446407
Height 0.154437
Whole weight 1.016849
Shucked weight 0.439051
Viscera weight 0.222526
Shell weight 0.291208
Table 1: Physical measurements for a new abalone.
3. Corresponding to the indices of the 3 nearest neighbors obtained above, extract the number of rings (ages) from
y_train. Among the 3 ages thus obtained, pick the age which appears most frequently (i.e., the mode). If multiple
ages appear equal number of times, pick one among them uniformly at random. The predicted age is the age we
will be assigning to new abalone data point. Report the predicted age.
4. Treating each row in X_test as a new abalone data point, repeat steps 13 above to generate a prediction for the
age. Store the predicted ages in an array called predicted_ages_test. Compute and report the mean squared
error (MSE) in prediction using the formula
(cid:18) (cid:19)
1 MX1 2
MSE = predicted_ages_test[j]  y_test[j] , (3)
M
j=0
where M is the length of y_test column.
2.5 Tuning k to Achieve Optimal Performance
Vary k from 1 to 50, and compute the MSE value for each k. Store the MSE values in an array named mse_values. Generate
a plot of k values on the xaxis and mse_values on the yaxis. From the plot, report an optimal value of k at which MSE is
the smallest. If there are multiple optimal values for k, you may pick any one at random. Also report the MSE corresponding
to this optimal value of k.

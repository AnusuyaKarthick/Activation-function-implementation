# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 06:53:26 2022

@author: KarthickAnu
"""
#load the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
data = pd.read_csv(r"C:\Users\KarthickAnu\Downloads\BankNote_Authentication.csv")

#No of records and columns in the dataset
data    #1372 rows and 5 columns
#column names
data.columns
#Top 10 rows and columns
data.head(10)
#missing values
data.isna().sum()   #No missing values

# show summary statistics
print(data.describe())

#split the dataset into Dependent and Independent variables
X = data.iloc[:,:4]
Y =data.iloc[:,4:5]
#shape of X and Y
X.shape
Y.shape
# ensure all data are floating point values
#X = X.astype('float32')
#Exploratory analysis
# plot histograms
data.hist()
plt.show()

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = X_train.T
y_train = np.reshape(y_train.to_numpy(),(1, y_train.shape[0]))
X_test = X_test.T
y_test = np.reshape(y_test.to_numpy(),(1, y_test.shape[0]))
print ("Train X Shape: ", X_train.shape)
print ("Train Y Shape: ", y_train.shape)
print ("I have m = %d training examples!" % (X_train.shape[1])) # 1097 training examples and 4 input features
print ('\nTest X Shape: ', X_test.shape)

#Define structure
def structure(X,Y):
   n_x = X.shape[0] # size of input layer
   n_h = 4 #1 hidden layer of size 4
   n_y = Y.shape[0] # size of output layer
   return (n_x, n_h, n_y)

(n_x, n_h, n_y) = structure(X_train, y_train)
print("No. of input layers ", n_x)
print("No. of hidden layers", n_h)
print("No. of output layers", n_y)

#Initialize weights and bias
def initialize_params(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    '''W3 = np.random.randn(n_h, n_x) * 0.01
    b3 = np.zeros((n_h, 1))'''

    return W1, b1, W2, b2

# activation function
def activation(k0,k1,x):
    return k0 + np.dot(k1,x)

from numpy import exp
def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid
def tanh(z):
    tan = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return tan
# softmax activation function

def softmax(activation):
    e = exp(activation)
    return e/e.sum()

#Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1  # x = a0, as activations of layer L0 be the input features of the training examples.
    A1 = activation(b1,W1,Z1)      #a1 =g(z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # for binary classification sigmoid activation function provide good accuracy.
    return Z1, A1, Z2, A2


#cost function
def compute_cost(Y, A2):

    m = Y.shape[1]
    cost = (-1/m) * np.sum(np.multiply(Y ,np.log(A2)) + np.multiply((1-Y), np.log(1-A2)))
    cost = float(np.squeeze(cost))
    return cost

#Backward propagation 
def backward_propagation(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2):
    
    m = X.shape[1] # Number of training examples

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * (np.sum(dZ2, axis = 1, keepdims = True)) #avg of elements of columns in the matrix dz2
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * (np.dot(dZ1, X.T))  #a0 =X
    db1 = (1/m) * (np.sum(dZ1, axis = 1, keepdims = True))  # avg of the elements in the column
    da1 =np.dot(W2.T,dZ2)
    daz = da1 * Z1
    avg_da1 = (1/m) *(np.sum(da1, keepdims=True))
    avg_daz = (1/m) *(np.sum(daz, keepdims=True))
    return dW1, db1, dW2, db2, da1, avg_da1, avg_daz


''' dk = dk1+dk2, here single layer hence only dk1
dk1 = matrix(avgE(da1),avgE(da1 *Z1),avgE(da1*Z1 power2)
where k = matrix(ko,k1,k2)'''

#update parameter weights and bias
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2
#neural network model
def neural_network(X, Y, n_h, learning_rate, num_iterations = 1000):
    n_x = structure(X, Y)[0]
    n_y = structure(X, Y)[2]
    costs = []
    # Initialize parameters
    W1, b1, W2, b2 = initialize_params(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(Y, A2)
        if i % 10 == 0:
            costs.append(cost)
        dW1, db1, dW2, db2,da1, avg_da1, avg_daz = backward_propagation(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    return W1, b1, W2, b2, costs, avg_da1, avg_daz


W1,b1,W2,b2,costs,avg_da1, avg_daz = neural_network(X_train, y_train, 4, 0.01, num_iterations=1000)

#k0=avg_da1 -   0.000214089, k1 = 0.0140542 =ava_daz
#Predictions
def predict(W1, b1, W2, b2, X):

    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)

    m = X.shape[1]
    y_pred = np.zeros((1, m))
    for i in range(A2.shape[1]):
        if A2[0, i] >= 0.5:
            y_pred[0, i] = 1
        else:
            y_pred[0, i] = 0

    return y_pred

#Train accuracy
predictions = predict(W1, b1, W2, b2, X_train)
train_accuracy = float((np.dot(y_train, predictions.T) + np.dot(1-y_train,1-predictions.T))/float(y_train.size) * 100)
print (f'Train_Accuracy: {round(train_accuracy, 2)} %')

#test accuracy
predictions = predict(W1, b1, W2, b2, X_test)
test_accuracy = float((np.dot(y_test, predictions.T) + np.dot(1-y_test,1-predictions.T))/float(y_test.size) * 100)
print (f'Test_Accuracy: {round(test_accuracy, 2)} %')
y_pred = predictions.T
y_test = y_test.T

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred )

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
print("Accuracy {:0.2f}%:".format(Accuracy))

#Precision 
Precision = tp/(tp+fp) 
print("Precision {:0.2f}".format(Precision))

Recall = tp/(tp+fn) 
print("Recall {:0.2f}".format(Recall))

#F1 Score
f1 = (2*Precision*Recall)/(Precision + Recall)
print("F1 Score {:0.2f}".format(f1))


#plot of loss function vs no. of iterations/epochs
plt.figure(figsize = (10, 6))
plt.title('loss Function')
plt.xlabel('No. of iterations/epochs')
plt.ylabel('Loss')
plt.plot(range(1, 1001, 10), costs)
plt.show() 

#Train and test loss
 
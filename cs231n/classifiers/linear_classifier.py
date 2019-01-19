from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None
      
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
#       temp_X = X.reshape(num_train*dim,1)
#       temp_y = y.T
      test_random = np.arange(0,num_train)
      random_number = np.random.choice(test_random,size = 500,replace = True)
      X_batch = X[random_number]
      y_batch = y[random_number]
      
      
      pass
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)
      
      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
     
      
      #W_temp = self.W
      
      self.W -= learning_rate*grad
#       print(self.W)
#       print(np.sum(W_temp)-np.sum(self.W))
    
#       loss_temp, grad_temp = self.loss(X_batch,y_batch,reg)
      
#       if(loss_temp < loss):
#         continue
      
      
      pass
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    scores = X.dot(self.W)
    
    for i in range(X.shape[0]):
        number = np.max(scores[i])
        #print(number)
        temp = scores[i].tolist()
        y_pred[i] = temp.index(number)
        
        
    pass
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function aWnd its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
#     num_train,dim = X_batch.shape
    
#     W_temp = self.W
#     scores = X_batch.dot(W_temp)
  
#     correct_scores = scores[np.arange(scores.shape[0]),y_batch]
    
#     correct_scores_T = correct_scores.reshape(num_train,1)
  
#     margins = np.maximum(0, scores - correct_scores_T + 1)
  
#     margins[np.arange(num_train),y_batch] = 0

#     loss = np.sum((np.sum(margins, axis=1)/num_train))
  
    
#     loss +=  reg * np.sum(W_temp * W_temp)
    
#     temp = margins.reshape(num_train,10)
#     temp[temp>0] = 1
#     number_of_margins_greater_than_zero_in_a_row = np.sum(temp,axis = 1, keepdims= True)
    
#     too_long_to_write_again = number_of_margins_greater_than_zero_in_a_row.reshape(1,num_train)
#     temp[np.arange(temp.shape[0]),y_batch] = -too_long_to_write_again
#     dW = X_batch.T.dot(temp)
#     dW /= num_train
#     dW += W_temp*2*reg
    
   
#     return loss,dW


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


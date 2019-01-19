import numpy as np
from random import shuffle
import math as m

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dim = X.shape[1]
  
  for i in range(num_train):
    
    scores = X[i].dot(W)
    
    correct_scores = scores[y[i]]
    margins = 0.0
    temp = np.exp(scores)
    for j in range(num_classes):
        if j == y[i]:
            dW[:,j] += -X[i]+ (X[i]*np.exp(W[:,j].T.dot(X[i])))/np.sum(temp)
            continue
        dW[:,j] += (X[i]*np.exp(W[:,j].T.dot(X[i]))/np.sum(temp)).T
        
        
#     temp = np.log(m.exp(correct_scores)/margins)
#     loss -= temp
        #temp = -correct_scores + np.log(margins)
    loss += -correct_scores + np.log(np.sum(temp))
  
  loss /= float(num_train)
  loss += float(reg)*np.sum(W**2)
  dW /= float(num_train)
  dW += 0.5*reg*W
  

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, num_dim = X.shape
  num_classes = W.shape[1]
  scores = X.dot(W)
  up = np.exp(scores[np.arange(num_train),y])
  down = np.sum(np.exp(scores),axis=1)
  #calculate loss
  loss = np.sum(-np.log(up/down))/num_train + reg*np.sum(W**2)
  

  X_T = X.T
  upper = np.exp(scores)
  bottom = np.sum(np.exp(scores), axis=1, keepdims=True)
  
  location_X = upper/bottom
  location_X[np.arange(num_train),y] -= 1
  #calculate dW
  dW = X_T.dot(location_X)
  dW /= num_train
  dW += 2*reg*W
  
  
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


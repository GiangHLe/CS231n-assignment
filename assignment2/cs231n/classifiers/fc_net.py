from builtins import range
from builtins import object
import numpy as np
import decimal as dm
from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        # add new dict have Xavier intialization std
        self.bonus = {}
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        # weight_scale = 1/np.sqrt(input_dim/2)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        # So, now we have new W with Xavier way
        #         self.bonus['w1_xavier'] = np.random.randn(input_dim, hidden_dim)/(np.sqrt(input_dim)
        #         self.bonus['w2_xavier'] = np.random.randn(hidden_dim, num_classes)/(np.sqrt(hidden_dim)
        #         self.params['W1'] = self.bonus['w1_xavier']
        #         self.params['W2'] = self.bonus['w2_xavier']
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        input = np.reshape(X, [X.shape[0], -1])
        hidden_layer, cache_1 = affine_relu_forward(X, W1, b1)
        scores, cache_2 = affine_relu_forward(hidden_layer, W2, b2)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        num_train = X.shape[0]
        up = np.exp(scores[np.arange(num_train), y])
        down = np.sum(np.exp(scores), axis=1)
        loss = np.sum(-np.log(up / down)) / num_train + 0.5 * self.reg * np.sum(W1 ** 2) + 0.5 * self.reg * np.sum(
            W2 ** 2)

        # not done
        upper = np.exp(scores)
        bottom = np.sum(upper, axis=1, keepdims=True)
        temp = upper / bottom
        # Calculate dz2, using double deviration
        temp[np.arange(temp.shape[0]), y] -= 1
        temp = temp / temp.shape[0]
        dout = temp

        d_input, d_W2, db2 = affine_relu_backward(dout, cache_2)
        d_x, d_W1, db1 = affine_relu_backward(d_input, cache_1)
        d_W1 += self.reg * W1
        d_W2 += self.reg * W2
        grads['W1'] = d_W1
        grads['W2'] = d_W2
        grads['b1'] = db1
        grads['b2'] = db2
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.weight_scale = weight_scale
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #                # this use in batch norm
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        self.length_hidden = len(hidden_dims)
        self.set_W_b(input_dim, hidden_dims[0], 1)
        if (normalization == 'batchnorm' or normalization == 'layernorm'):
            self.set_gamma_beta(hidden_dims[0], 1)
        if (self.length_hidden > 1):
            for ID in range(self.length_hidden-1):
                # use setdefault because maybe this is 1 layer net,the weight has already created above
                self.set_W_b(hidden_dims[ID], hidden_dims[ID+1], (ID+2))
                if (normalization == 'batchnorm' or normalization == 'layernorm'):
                    self.set_gamma_beta(hidden_dims[ID+1], (ID+2))
        self.set_W_b(hidden_dims[self.length_hidden-1], num_classes, (self.length_hidden+1))             # for final layer
        # if (normalization == 'batchnorm' or normalization == 'layernorm'):
        #     self.set_gamma_beta(hidden_dims[self.length_hidden-1], (self.length_hidden +1))
        # for a,b in self.params:
        #     print(a,b)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
            self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

            # Cast all parameters to the correct datatype
            for k, v in self.params.items():
                self.params[k] = v.astype(dtype)
    # set param W
    def set_W_b(self, row,column,ID):
        '''
        Set weight for fc layer
        '''
        w_temp = self.weight_scale * np.random.randn(row,column)
        b_temp = np.zeros(column)
        name_of_weight = 'W' + str(ID)
        name_of_bias = 'b' + str(ID)
        self.params.setdefault(name_of_weight, w_temp)
        self.params.setdefault(name_of_bias, b_temp)

    def set_gamma_beta(self, column,ID):
        '''
        Set parameter for batchnorm layer
        '''
        name_of_gamma = 'gamma' + str(ID)# set gamma name
        name_of_beta = 'beta' + str(ID)  # set beta name
        gamma = np.ones(column)  
        beta = np.zeros(column)
        self.params.setdefault(name_of_gamma, gamma)
        self.params.setdefault(name_of_beta, beta)
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = {}
        forward = {'0': X}
        sum_W = 0
        ln_param = {}
        for i in range(self.length_hidden+1):
            name_of_W = 'W' + str(i + 1)
            name_of_b = 'b' + str(i + 1)
            name_of_gamma = 'gamma' + str(i + 1)
            name_of_beta = 'beta' + str(i + 1)
            W = self.params[name_of_W]
            b = self.params[name_of_b]
            sum_W += np.sum(W**2)
            if i < self.length_hidden:
                if self.use_dropout and (self.normalization == 'batchnorm'):
                    gamma = self.params[name_of_gamma]
                    beta = self.params[name_of_beta]
                    bn_params = self.bn_params[i]
                    forward[str(i+1)], cache[str(i)] = affine_batchnorm_relu_dropout_forward(forward[str(i)], W, b, gamma, beta,
                                                                                    bn_params, self.dropout_params)
                elif self.use_dropout and (self.normalization == 'layernorm'):
                    gamma = self.params[name_of_gamma]
                    beta = self.params[name_of_beta]
                    forward[str(i+1)], cache[str(i)] = affine_layernorm_relu_dropout_forward(forward[str(i)], W, b, gamma, beta,
                                                                                    ln_param,self.dropout_param)
                elif (self.normalization == 'layernorm'):
                    gamma = self.params[name_of_gamma]
                    beta = self.params[name_of_beta]
                    forward[str(i + 1)], cache[str(i)] = affine_layernorm_relu_forward(forward[str(i)], W, b, gamma, beta, ln_param)
                elif (self.normalization == 'batchnorm'):
                    gamma = self.params[name_of_gamma]
                    beta = self.params[name_of_beta]
                    bn_params = self.bn_params[i]
                    forward[str(i + 1)], cache[str(i)] = affine_batchnorm_relu_forward(forward[str(i)], W, b, gamma, beta, bn_params)
                elif (self.use_dropout):
                    forward[str(i + 1)], cache[str(i)] = affine_relu_dropout_forward(forward[str(i)], W, b, self.dropout_param)
                else:
                    forward[str(i + 1)], cache[str(i)] = affine_relu_forward(forward[str(i)], W, b)
            else:
                scores, cache[str(i)] = affine_forward(forward[str(i)], W, b)


        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        '''
        Compute Loss
        '''
        dm.getcontext().prec = 100
        temp = scores[np.arange(scores.shape[0]), y]
        
        upper = np.exp(temp)
        bottom = np.sum(np.exp(scores), axis=1)
        loss = np.sum(-np.log(upper / bottom)) / scores.shape[0] + 0.5 * self.reg * sum_W
        '''
        Compute gradient
        '''
        backgrop = {}
        top = np.exp(scores)
        bot = np.sum(top, axis =1, keepdims = True)
        dout = top / bot
        dout[np.arange(dout.shape[0]), y] -= 1
        dout /= scores.shape[0]
        dX, dw, db = affine_backward(dout, cache[str(self.length_hidden)])
        w_last = self.params['W'+str(self.length_hidden+1)]
        dw += self.reg * w_last
        backgrop[str(self.length_hidden)] = dX
        grads['W'+str(self.length_hidden+1)] = dw
        grads['b'+str(self.length_hidden+1)] = db
        a = self.length_hidden
        '''
        From hidden last - 1 to input
        '''
        while(a>0):
            if self.use_dropout and (self.normalization == 'batchnorm'):
                backgrop[str(abs(a-1))], dW, grads['b'+str(a)] , grads['gamma' + str(a)], grads['beta' + str(a)]= affine_batchnorm_relu_dropout_backward(backgrop[str(a)], cache[str(a-1)])
                w = self.params['W' + str(a)]
                dW += self.reg * w
                grads['W' + str(a)] = dW

            elif self.use_dropout and (self.normalization == 'layernorm'):
                backgrop[str(abs(a-1))], dW, grads['b'+str(a)] , grads['gamma' + str(a)], grads['beta' + str(a)]= affine_layernorm_relu_dropout_backward(backgrop[str(a)], cache[str(a-1)])
                w = self.params['W' + str(a)]
                dW += self.reg * w
                grads['W' + str(a)] = dW

            elif (self.normalization == 'batchnorm'):
                backgrop[str(abs(a-1))], dW, grads['b'+str(a)] , grads['gamma' + str(a)], grads['beta' + str(a)] = affine_batchnorm_relu_backward(backgrop[str(a)], cache[str(a-1)])
                w = self.params['W' + str(a)]
                dW += self.reg * w
                grads['W' + str(a)] = dW

            elif (self.normalization == 'layernorm'):
                backgrop[str(abs(a-1))], dW, grads['b'+str(a)] , grads['gamma' + str(a)], grads['beta' + str(a)] = affine_layernorm_relu_backward(backgrop[str(a)], cache[str(a-1)])
                w = self.params['W' + str(a)]
                dW += self.reg * w
                grads['W' + str(a)] = dW
            elif (self.use_dropout):
                backgrop[str(abs(a-1))], dW, grads['b'+str(a)] = affine_relu_dropout_backward(backgrop[str(a)], cache[str(a-1)])
                w = self.params['W' + str(a)]
                dW += self.reg * w
                grads['W' + str(a)] = dW

            else:
                backgrop[str(abs(a-1))], dW, grads['b'+str(a)] = affine_relu_backward(backgrop[str(a)], cache[str(a-1)])
                w = self.params['W' + str(a)]
                dW += self.reg * w
                grads['W' + str(a)] = dW
            a-=1
        pass
        # for key, value in self.params.items():
        #     print ("%s key has the value %s" % (key, value))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads

import numpy as np
import math as math

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = np.shape(X)

    for i in range(N):
        xw = 0;    # y_pred at ith example
        for feature in range(D):
            xw += X[i][feature] * W[feature]

        loss += (xw - y[i])**2

        for feature in range(D):
            dW[feature] += (xw - y[i]) * X[i][feature]    # compute derivative over WEIGHTS

    reg_loss = 0
    for feature in range(D):
        reg_loss = reg_loss + W[feature]**2
        dW[feature] = (dW[feature] + reg * W[feature]) / N

    loss = (loss + reg_loss * reg) / (2 * N)

    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N, D = np.shape(X)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    y_pred = np.dot(X, W)
    y = np.reshape(y, np.shape(y_pred))    # notice (R, 1) and (R,) !!!!
    diff = y_pred - y
    loss = (np.sum(diff ** 2) + reg * np.sum(W ** 2)) / (2 * N)
    dW = (np.dot(np.transpose(X), diff) + reg * W) / N

    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
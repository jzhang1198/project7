# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import pytest
from nn import NeuralNetwork
from numpy.typing import ArrayLike
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# TODO: Write your test functions and associated docstrings below.

def load_test_data(split_percent=None):
    """
    Loads digits dataset from sklearn for subsequent unit testing.

    Args:
        split_percent: float
            Percentage of dataset to be partitioned into validation set.

    Returns:
        X_train: ArrayLike
            Observations for training.
        y_train: ArrayLike:
            Labels for training.
        X_val:
            Observations for validation.
        y_val:
            Labels for validation.
    """

    #Load and split digits dataset

    digits = load_digits()
    y = digits['target']
    X = digits['data']
    return X.T, y.T

    if split_percent is not None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_percent, random_state=42)

        #Reshape dataset to fit convention used in implementation of neural net
        X_train = X_train.T
        X_val = X_val.T
        y_train = np.array([y_train]).T
        y_val = np.array([y_val]).T
        return X_train, y_train, X_val, y_val

def instantiate_nn(lr: float, batch_size: int, epochs: int, loss_function: str):
    """
    Helper function to instantiate a NeuralNetwork object for unit testing.
    """
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'}, {'input_dim': 16, 'output_dim': 64, 'activation:': 'sigmoid'}]
    seed = 1
    return NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

def test_forward_and_single_forward_and_predict():
    """
    Tests forward pass through the network.
    """
    X, y = load_test_data() #load dataset

    lr = 0.0001
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'mse'
    nn = instantiate_nn(lr, batch_size, 1, loss_function)

    A_prev = X
    for i in range(0,len(self.arch)):
        act_func_type = self.arch[i]['activation']
        W_curr = self._param_dict['W' + str(i+1)]
        b_curr = self._param_dict['b' + str(i+1)]
        assert W_curr.shape[0] == b_curr.shape[0] #double check that W and b are constructed correctly

        Z_curr, A_curr = self._single_forward(W_curr, b_curr, A_prev, act_func_type)
        assert A_curr.shape = Z_curr.shape #check that dimensions of A and Z are correct
        assert A_curr.shape[1] == X.shape[1] and A_curr.shape[0] == W_curr.shape[0]

        #double check that the columns and rows are dotted properly
        for j in range(0, W_curr.shape[0]):
            b = b_curr[j]
            for k in range(0, A_prev.shape[1]):
                assert np.dot(W_curr[j, :], A_prev[:, k]) + b - Z_curr[j,k] < 10e-8

def test_backprop_and_single_backprop():
    X, y = load_test_data() #load dataset

    lr = 0.0001 #instantiate nn
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'mse'
    nn = instantiate_nn(lr, batch_size, 1, loss_function)

    y_hat, cache = nn.forward(X) #forward pass of X

    grad_dict = {}

    #compute the gradients for layer L
    activation_curr = self.arch[-1]['activation']
    Z_curr = cache['Z' + str(len(self.arch))]
    delta_curr = np.multiply(self._loss_backprop(y, y_hat, self._loss_func), self._activation_backprop(Z_curr, activation_curr)) #compute the product of dJ/dAL and dAL/dZl
    assert delta_curr.shape == y.shape and delta_curr.shape == y_hat.shape and delta_curr.shape == Z_curr.shape #check that the shape of delta is correct
    assert delta_curr.shape[1] == X.shape[1]

    A_prev = cache['A' + str(len(self.arch)-1)]

    #update grad_dict
    grad_dict['dW' + str(len(self.arch))] = np.dot(delta_curr, A_prev.transpose())
    grad_dict['db' + str(len(self.arch))] = np.sum(delta_curr, axis=1, keepdims=True)
    assert grad_dict['dW' + str(len(self.arch))].shape == self._param_dict['W' + str(len(self.arch))].shape #check that the shapes of gradients are correct
    assert grad_dict['db' + str(len(self.arch))].shape == self._param_dict['b' + str(len(self.arch))].shape

    delta_prev = delta_L
    for i in range(0,len(self.arch)-1)[::-1]:
        activation_curr = self.arch[i]['activation']
        W_curr = self._param_dict['W' + str(i+2)]
        b_curr = self._param_dict['b' + str(i+2)]
        Z_curr = cache['Z' + str(i+1)]
        A_prev = cache['A' + str(i)]
        assert W_curr.shape[1] == Z_curr.shape[1] #check that the shapes of cached matrices are correct
        assert b_curr.shape[0] == W_curr.shape[0]

        delta_curr = self._compute_delta(W_curr, delta_prev, Z_curr, activation_curr) #compute delta
        assert delta_curr.shape == Z_curr.shape #check that the shape of delta is correct

        #compute gradients
        dW_curr, db_curr = self._single_backprop(A_prev, delta_curr)

        #update grad_dict and delta
        grad_dict['dW' + str(i+1)] = dW_curr
        grad_dict['db' + str(i+1)] = db_curr
        delta_prev = delta_curr
        assert grad_dict['dW' + str(i+1)].shape == self._param_dict['W' + str(i+1)].shape #check that the shape of gradients are correct
        assert grad_dict['db' + str(i+1)].shape == self._param_dict['b' + str(i+1)].shape

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass

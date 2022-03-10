# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import pytest
from nn import NeuralNetwork
from nn import preprocess
from numpy.typing import ArrayLike
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from nn import io

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
    return X.T, np.array([y])

    if split_percent is not None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_percent, random_state=42)

        #Reshape dataset to fit convention used in implementation of neural net
        X_train = X_train.T
        X_val = X_val.T
        y_train = np.array([y_train])
        y_val = np.array([y_val])
        return X_train, y_train, X_val, y_val

def instantiate_nn(nn_arch, lr: float, batch_size: int, epochs: int, loss_function: str):
    """
    Helper function to instantiate a NeuralNetwork object for unit testing.
    """
    seed = 1 #set seed for reproducibility
    return NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

def test_forward_and_single_forward_and_predict():
    """
    Tests forward pass through the network.
    """
    X, y = load_test_data() #load dataset

    #instantiate a NeuralNetwork object
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}]
    lr = 0.0001
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'mse'
    nn = instantiate_nn(nn_arch, lr, batch_size, 1, loss_function)

    #copy of source code from forward method
    A_prev = X
    for i in range(0,len(nn.arch)):
        act_func_type = nn.arch[i]['activation']
        W_curr = nn._param_dict['W' + str(i+1)]
        b_curr = nn._param_dict['b' + str(i+1)]
        assert W_curr.shape[0] == b_curr.shape[0] #double check that W and b are constructed correctly

        Z_curr, A_curr = nn._single_forward(W_curr, b_curr, A_prev, act_func_type)
        assert A_curr.shape == Z_curr.shape #check that dimensions of A and Z are correct
        assert A_curr.shape[1] == X.shape[1] and A_curr.shape[0] == W_curr.shape[0]

        #double check that the columns and rows are dotted properly
        for j in range(0, W_curr.shape[0]):
            b = b_curr[j]
            for k in range(0, A_prev.shape[1]):
                assert np.dot(W_curr[j, :], A_prev[:, k]) + b - Z_curr[j,k] < 10e-8

        A_prev = A_curr

def test_backprop_and_single_backprop():
    X, y = load_test_data() #load dataset

    #instantiate a NeuralNetwork object
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}]
    lr = 0.0001
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'mse'
    nn = instantiate_nn(nn_arch, lr, batch_size, 1, loss_function)

    y_hat, cache = nn.forward(X) #forward pass of X

    #copy of source code from backprop method:
    grad_dict = {}

    #compute the gradients for layer L
    activation_curr = nn.arch[-1]['activation']
    Z_curr = cache['Z' + str(len(nn.arch))]
    delta_curr = np.multiply(nn._loss_backprop(y, y_hat, nn._loss_func), nn._activation_backprop(Z_curr, activation_curr)) #compute the product of dJ/dAL and dAL/dZl
    assert delta_curr.shape == Z_curr.shape #check that the shape of delta is correct
    assert delta_curr.shape[1] == X.shape[1]

    A_prev = cache['A' + str(len(nn.arch)-1)]

    #update grad_dict
    grad_dict['dW' + str(len(nn.arch))] = np.dot(delta_curr, A_prev.transpose())
    grad_dict['db' + str(len(nn.arch))] = np.sum(delta_curr, axis=1, keepdims=True)
    assert grad_dict['dW' + str(len(nn.arch))].shape == nn._param_dict['W' + str(len(nn.arch))].shape #check that the shapes of gradients are correct
    assert grad_dict['db' + str(len(nn.arch))].shape == nn._param_dict['b' + str(len(nn.arch))].shape

    delta_prev = delta_curr
    for i in range(0,len(nn.arch)-1)[::-1]:
        activation_curr = nn.arch[i]['activation']
        W_curr = nn._param_dict['W' + str(i+2)]
        b_curr = nn._param_dict['b' + str(i+2)]
        Z_curr = cache['Z' + str(i+1)]
        A_prev = cache['A' + str(i)]
        assert W_curr.shape[1] == Z_curr.shape[0] #check that the shapes of cached matrices are correct
        assert b_curr.shape[0] == W_curr.shape[0]

        delta_curr = nn._compute_delta(W_curr, delta_prev, Z_curr, activation_curr) #compute delta
        assert delta_curr.shape == Z_curr.shape #check that the shape of delta is correct

        #compute gradients
        dW_curr, db_curr = nn._single_backprop(A_prev, delta_curr)

        #update grad_dict and delta
        grad_dict['dW' + str(i+1)] = dW_curr
        grad_dict['db' + str(i+1)] = db_curr
        delta_prev = delta_curr
        assert grad_dict['dW' + str(i+1)].shape == nn._param_dict['W' + str(i+1)].shape #check that the shape of gradients are correct
        assert grad_dict['db' + str(i+1)].shape == nn._param_dict['b' + str(i+1)].shape

def test_binary_cross_entropy():
    X, y = load_test_data() #load dataset

    #instantiate a NeuralNetwork object
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 0.0001
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'bce'
    nn = instantiate_nn(nn_arch, lr, batch_size, 1, loss_function)

    y_hat, _ = nn.forward(X)
    y = nn._sigmoid(y) #map y to floats between 0 and 1
    loss = nn._loss_function(y, y_hat, nn._loss_func)

    assert type(loss) == float or type(loss) == np.float64 #check that data type of output is correct
    assert loss > 0 #check that loss is reasonable

def test_binary_cross_entropy_backprop():
    X, y = load_test_data() #load dataset

    #instantiate a NeuralNetwork object
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 0.0001
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'bce'
    nn = instantiate_nn(nn_arch, lr, batch_size, 1, loss_function)

    y_hat, cache = nn.forward(X)
    y = nn._sigmoid(y) #map y to floats between 0 and 1
    dA = nn._loss_backprop(y, y_hat, nn._loss_func)

    assert dA.shape == y_hat.shape #check that the shape of bce backprop is reasonable
    assert dA.shape == y.shape

def test_mean_squared_error():
    X, y = load_test_data() #load dataset

    #instantiate a NeuralNetwork object
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}]
    lr = 0.0001
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'mse'
    nn = instantiate_nn(nn_arch, lr, batch_size, 1, loss_function)

    y_hat, _ = nn.forward(X)
    loss = nn._loss_function(y, y_hat, nn._loss_func)

    assert type(loss) == float or type(loss) == np.float64 #check that data type of output is correct
    assert loss > 0 #check that loss is reasonable

def test_mean_squared_error_backprop():
    X, y = load_test_data() #load dataset

    #instantiate a NeuralNetwork object
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}]
    lr = 0.0001 #instantiate nn
    batch_size = X.shape[1]
    epochs = 1
    loss_function = 'mse'
    nn = instantiate_nn(nn_arch, lr, batch_size, 1, loss_function)

    y_hat, cache = nn.forward(X)
    dA = nn._loss_backprop(X, y_hat, nn._loss_func)

    assert dA.shape == y_hat.shape #check that the shape of mse backprop is reasonable
    assert dA.shape == X.shape

def test_one_hot_encode():
    seq_arr = ['ATCG','AAAA','TTTT','CCCC','GGGG'] #generate test sequences
    encodings = preprocess.one_hot_encode_seqs(seq_arr)

    assert type(encodings) == list #check that encodings were constructed correctly
    for seq in encodings:
        assert type(seq) == list
        assert set(seq) == {0,1}

def test_sample_seqs():
    #Process data for training
    pos_file = 'data/rap1-lieb-positives.txt'
    neg_file = 'data/yeast-upstream-1k-negative.fa'

    #Load sequences and labels
    pos_seqs = io.read_text_file(pos_file)
    pos_labels = [1] * len(pos_seqs)
    neg_seqs = io.read_fasta_file(neg_file)
    neg_labels = [0] * len(neg_seqs)
    seqs = pos_seqs + neg_seqs
    labels = pos_labels + neg_labels

    #Sample sequences
    seed = 1 #set seed for reproducibility
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs,labels,1)

    #check that sampled_seqs and sampled_labels were constructed correctly
    assert set(sampled_labels) == {0,1}
    assert len([i for i in sampled_labels if i == 0]) == len([i for i in sampled_labels if i == 1])
    assert type(sampled_labels) == list
    assert type(sampled_seqs) == list

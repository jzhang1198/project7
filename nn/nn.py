#BMI 203 Project 7: Neural Network


#Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch,
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self, W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix. The ith row of W_curr represents the weights corresponding to the ith neuron of the next layer.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix. The columns of A_prev are indexed by the observations.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        Z_curr = np.dot(W_curr, A_prev) + b_curr
        A_curr = self._activation_function(Z_curr, activation)
        return Z_curr, A_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [features, batch_size].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        cache = {}

        # print(type(X))
        A_prev = X
        # print(type(A_prev))
        cache['A0'] = A_prev
        # print(type(cache['A0']))
        for i in range(0,len(self.arch)):
            # print(i)
            act_func_type = self.arch[i]['activation'] #get W, b, and the activation function type for layer i+1
            W_curr = self._param_dict['W' + str(i+1)]
            b_curr = self._param_dict['b' + str(i+1)]

            # print(type(A_prev), type(W_curr), type(b_curr))
            # print(A_prev.shape,W_curr.shape, b_curr.shape)

            Z_curr, A_curr = self._single_forward(W_curr, b_curr, A_prev, act_func_type) #compute Z and A for layer i+1
            cache['A' + str(i+1)] = A_curr #cache output Z and A from layer i+1
            cache['Z' + str(i+1)] = Z_curr

            A_prev = A_curr #update A_prev for next iteration

        output = A_prev

        return output, cache

    def _single_backprop(self,
                         A_prev: ArrayLike,
                         delta_curr: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            A_prev: ArrayLike
                Previous layer activation matrix.
            delta_curr: ArrayLike
                Delta for current layer.

        Returns:
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        #compute gradients for biases and weights
        db_curr = np.sum(delta_curr, axis=1, keepdims=True)
        dW_curr = np.dot(delta_curr, A_prev.transpose())
        return dW_curr, db_curr

    def _compute_delta(self, W_curr: ArrayLike, delta_prev: ArrayLike, Z_curr: ArrayLike, activation_curr: str):
        """
        Computes delta, as defined in https://sudeepraja.github.io/Neural/, for the current layer.

        Args:
            W_curr: ArrayLike
                Weight matrix for the current layer.
            delta_prev: ArrayLike
                Delta quantity from previous layer.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            activation_curr: str
                Activation function (sigmoid or relu).

        Returns:
            delta: ArrayLike
                Delta quantity for current layer.
        """

        dZ = self._activation_backprop(Z_curr, activation_curr) #for layer i, compute derivative of activation function with respect to Zi
        delta = np.multiply(np.dot(W_curr.transpose(), delta_prev), dZ) #for layer i, compute the product of the dJ/dZi+1, dZi+1/dAi, and dAi/dZi
        return delta

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        grad_dict = {}

        #compute the gradients for layer L
        activation_curr = self.arch[-1]['activation']
        Z_curr = cache['Z' + str(len(self.arch))]
        delta_L = np.multiply(self._loss_backprop(y, y_hat, self._loss_func), self._activation_backprop(Z_curr, activation_curr)) #compute the product of dJ/dAL and dAL/dZl
        A_prev = cache['A' + str(len(self.arch)-1)]

        #update grad_dict
        grad_dict['dW' + str(len(self.arch))] = np.dot(delta_L, A_prev.transpose())
        grad_dict['db' + str(len(self.arch))] = np.sum(delta_L, axis=1, keepdims=True)

        delta_prev = delta_L
        for i in range(0,len(self.arch)-1)[::-1]:
            activation_curr = self.arch[i]['activation']
            W_curr = self._param_dict['W' + str(i+2)]
            b_curr = self._param_dict['b' + str(i+2)]
            Z_curr = cache['Z' + str(i+1)]
            A_prev = cache['A' + str(i)]

            delta_curr = self._compute_delta(W_curr, delta_prev, Z_curr, activation_curr) #compute delta

            #compute gradients
            dW_curr, db_curr = self._single_backprop(A_prev, delta_curr)

            #update grad_dict and delta
            grad_dict['dW' + str(i+1)] = dW_curr
            grad_dict['db' + str(i+1)] = db_curr
            delta_prev = delta_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """

        for key in grad_dict:
            grad = grad_dict[key] #get gradient and current weights
            current_weights = self._param_dict[key[1:]]
            self._param_dict[key[1:]] = current_weights - self._lr * grad #update weights

        return

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        epochs = 0
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        assert type(epochs) == int
        assert type(self._epochs) == int
        while epochs <= self._epochs:
            epochs += 1
            assert type(epochs) == int

            #Shuffle training set
            shuffled_indices = np.linspace(0, X_train.shape[1]-1, X_train.shape[1])
            np.random.shuffle(shuffled_indices)
            X_train = np.array([X_train[:, int(i)] for i in shuffled_indices]).T
            y_train = np.array([y_train[:, int(i)] for i in shuffled_indices]).T

            #Split training set into batches
            num_batches = int(X_train.shape[1]/self._batch_size) + 1
            X_batch = [i.T for i in np.array_split(X_train.T, num_batches)]
            y_batch = [i.T for i in np.array_split(y_train.T, num_batches)]

            #Generate empty lists to hold losses for batches
            training_losses = []
            validation_losses = []

            for X_train, y_train in zip(X_batch, y_batch):
                assert type(X_train) == np.ndarray
                training_output, training_cache = self.forward(X_train) #forward pass of training and validation set
                val_output, val_cache = self.forward(X_val)

                training_losses.append(self._loss_function(y_train, training_output, self._loss_func)) #record training and validation losses
                validation_losses.append(self._loss_function(y_val, val_output, self._loss_func))

                grad_dict = self.backprop(y_train, training_output, training_cache) #backward pass
                self._update_params(grad_dict) #update weights and biases

            per_epoch_loss_train.append(np.mean(np.array(training_losses)))
            per_epoch_loss_val.append(np.mean(np.array(validation_losses)))

        return per_epoch_loss_train, per_epoch_loss_val



    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)

        return y_hat

    def _activation_function(self, Z: ArrayLike, activation: str) -> ArrayLike:
        """
        Applies activation function to input Z array.

        Args:
            Z: ArrayLike
                Output of layer linear transform.
            activation: str
                Type of activation function (either sigmoid or relu)

        Returns:
            A: ArrayLike
                Activation matrix.
        """

        if activation == 'sigmoid':
            A = self._sigmoid(Z)

        elif activation == 'relu':
            A = self._relu(Z)

        return A

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0,Z)
        return nl_transform

    def _activation_backprop(self, Z: ArrayLike, activation: str) -> ArrayLike:
        """
        Applies derivative of activation function to an input A array.

        Args:
            Z: ArrayLike
                Output of layer linear transform
            activation: str
                Type of activation function (either sigmoid or relu)

        Returns:
            dZ: ArrayLike
                Derivative of loss function J with respect to Z.
        """

        if activation == 'sigmoid':
            dZ = self._sigmoid_backprop(Z)

        elif activation == 'relu':
            dZ = self._relu_backprop(Z)

        return dZ

    def _sigmoid_backprop(self, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        dZ = self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dZ

    def _relu_backprop(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        dZ = self._relu(Z)
        dZ[dZ > 0] = 1
        return dZ

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike, loss_fun: str) -> float:
        """
        Wrapper function for loss functions.

        Args:
            y: ArrayLike
                Ground truth labels for data.
            y_hat: ArrayLike
                Model predictions for labels.
            loss_fun: str
                Type of loss function (bce, mse, or name of user defined loss function)

        Returns:
            loss: float
                The loss.
        """

        if loss_fun == 'bce':
            loss = self._binary_cross_entropy(y, y_hat)

        elif loss_fun == 'mse':
            loss = self._mean_squared_error(y, y_hat)

        return loss

    def _loss_backprop(self, y: ArrayLike, y_hat: ArrayLike, loss_fun: str) -> ArrayLike:
        """
        This function computes dJ/dAL for a given loss function.
        Args:
            y: ArrayLike
                Ground truth.
            y_hat: ArrayLike
                Neural net predictions.
            loss_fun: str
                Type of loss function (either bce or mse)

        Returns:
            dA: ArrayLike
                Partial derivative of the loss function with respect to activation matrix
        """

        if loss_fun == 'bce':
            dA = self._binary_cross_entropy_backprop(y, y_hat)

        elif loss_fun == 'mse':
            dA = self._mean_squared_error_backprop(y, y_hat)

        return dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """

        loss = -1 * np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1 - y, np.log(1 - y_hat))) / y.shape[1]
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        dA = -1 * np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat) / y.shape[1]
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """

        loss = np.mean(np.square(y - y_hat), axis=0)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        dA = (-2 / (y.size)) * (y - y_hat)

        return dA

    def _user_loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """

        pass

    def _user_loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass

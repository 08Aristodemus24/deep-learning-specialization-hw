import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from C1_W3.planar_utils import load_extra_datasets, load_planar_dataset, plot_decision_boundary
# from C1_W4_2nd_Assignment.dnn_app_utils_v3 import load_data

class NeuralNetwork:
    def __init__(self, 
                 X, 
                 Y, 
                 layers_dims=[2, 15, 15, 3, 1], 
                 epochs=5000, 
                 rec_ep_at=500,
                 learning_rate=0.01, 
                 keep_prob=0.7, 
                 initializer='xavier', 
                 hl_acts="relu", 
                 ol_acts="sigmoid", 
                 loss="binary_cross_entropy", 
                 regularization=None,
                 lambda_=0.1) -> None:
        # input and output values
        self.X = X
        self.Y = Y

        # dictionary of all coefficients
        # THETA_0, BETA_0, THETA_1, BETA_1, THETA_2, BETA_2, ..
        self.params = {}

        # type of initializer of parameters
        self.initializer = initializer

        # architecture (arc) defines the number layers and number of nodes in each layer
        self.layers_dims = layers_dims
        self.L = len(layers_dims)

        # indication of what activation to use for
        # the hidden layers and output layer
        self.hl_acts = hl_acts
        self.ol_acts = ol_acts

        # indication of what loss to use to 
        # solve a type of problem
        self.loss = loss

        # hyper params
        self.epochs = epochs\
        
        # records epoch at "rec_ep_at" e.g. records epoch at 500
        self.rec_ep_at = rec_ep_at
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.num_features = X.shape[0]
        self.num_instances = X.shape[1]
        self.regularization = regularization
        self.lambda_ = lambda_

        self._cost_per_iter = []


    def init_params(self):

        if self.initializer.lower() == "he":
            multiplier = lambda layer: np.sqrt(2.0 / self.layers_dims[layer - 1])
        elif self.initializer.lower() == "xavier":
            multiplier = lambda layer: np.sqrt(1.0 / self.layers_dims[layer - 1])
        elif self.initializer.lower() == "lg-rand":
            multiplier = lambda layer: 10
        elif self.initializer.lower() == "sm-rand":
            multiplier = lambda layer: 0.01

        np.random.seed(3)

        # build dictionary of coefficients
        for layer in range(1, self.L):
            self.params[f'THETA_{layer - 1}'] = np.random.randn(self.layers_dims[layer], self.layers_dims[layer - 1]) * multiplier(layer)
            self.params[f'BETA_{layer - 1}'] = np.zeros((self.layers_dims[layer], 1)) * multiplier(layer)

    def flatten_params(self):
        pass

    def flatten_grads(self):
        pass

    @property
    def costs_per_iter(self):
        return list(zip(*self._costs_per_iter))

    def view_cost(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        costs, epochs = self.costs_per_iter

        ax.plot(costs, epochs, 'p:', c='#5d42f5', label='cost')
        ax.legend()
        plt.show()
        

    def train(self):
        # initialize THETAs and BETAs
        self.init_params()
        print(self.params)

        # run for a number of epochs
        for epoch in range(self.epochs):
            
            # forward and save dot products and activations of each layer
            caches = self.forward(self.X)

            # calculate error
            if epoch % self.rec_ep_at == 0:
                cost = self.J(caches)
                self._cost_per_iter.append((cost, epoch))
                print(f'cost: {cost}\n')

            # backward propagate to calculate gradients
            grads = self.backward(caches)
            
            # update current THETAs and BETAs using update rule and gradients
            self.update(grads)


    def J(self, caches):
        # get activation of last layer A_L - 1
        A_last = caches[f'A_{self.L - 1}']

        if self.loss.lower() == "binary_cross_entropy":
            # Y and A_last both have shape 1 x 280
            loss = np.dot(-1.0 * self.Y, np.log(A_last).T) - np.dot((1 - self.Y), np.log(1 - A_last).T)
            cost = (np.squeeze(loss) / self.num_instances) + self.regularizer()
        
        elif self.loss.lower() == "mean_squared_error":
            loss = np.dot(A_last - self.Y, (A_last - self.Y).T)
            cost = loss / (2 * self.num_instances)

        return cost

    def L_prime_A_last(self, caches):
        try:
            A_last = caches[f'A_{self.L - 1}']
            # print(A_last)

            if self.loss.lower() == "binary_cross_entropy":
                dloss = (-1 * np.divide(self.Y, A_last)) + np.divide(1 - self.Y, 1 - A_last)

            elif self.loss.lower() == "mean_squared_error":
                dloss = A_last - self.Y

            return dloss         

        except RuntimeWarning as e:
            print(f'error is: {e}')

            print(self.Y, A_last)
        
    
    def regularizer(self):
        if self.regularization == None:
            return 0.0
        
        elif self.regularization.upper() == "L2":
            
            # get the square of all coefficients in each layer excluding biases
            # if 5 layers then loop from 0 to 3 to access all coefficients
            l2_norm = 0 

            # if there is only 2 layers then calculation
            # in loop only runs once
            for layer in range(self.L - 1):
                # take teh squared value of each value in coefficient matrix
                squared_coeffs = np.square(self.params[f'THETA_{layer}'])

                # sum all values in the coefficient matrix
                summed_coeffs = np.sum(squared_coeffs)
                l2_norm += summed_coeffs

            # multiply by lambda constant
            l2_norm = (self.lambda_ * l2_norm) / (2 * self.num_instances)
            return l2_norm

        elif self.regularization.upper() == "L1":
            # if there is only 2 layers then calculation
            # in loop only runs once
            l1_norm = 0
            for layer in range(self.L - 1):
                # take the absolute value of each in coefficient matrix
                abs_coeffs = np.abs(self.params[f'THETA_{layer}'])

                # sum all values in the coefficient matrix
                summed_coeffs = np.sum(abs_coeffs)
                l1_norm += summed_coeffs

            l1_norm = (self.lambda_ * l1_norm) / (2 * self.num_instances)
            return l1_norm

    def regularizer_prime(self):
        if self.regularization == None:
            return 0.0
        
        elif self.regularization.upper() == "L2":
            return 0.0
        
        elif self.regularization.upper() == "L1":
            return 0.0

    def forward(self, X):
        # limit L since output layer has different activation
        # hence loop only until 2nd to the last layer e.g. 4 - 1 -> [1 - 1, 2 - 1]
        L = self.L - 1 

        # set A or A_0 to input X put 
        # also X in cache dict
        caches = {}
        A = X
        caches[f'A_0'] = A

        # loop stops at 2nd to the last layer/output layer since
        # output has different activation
        for layer in range(1, L):
            Z = self.linear(A, self.params[f'THETA_{layer - 1}'], self.params[f'BETA_{layer - 1}'])
            A = self.g(Z, self.hl_acts)
            
            # build or update dictionary of caches from A_1, ... A_L - 1
            # and from Z_1, ..., Z_L - 1 
            caches[f'Z_{layer}'] = Z
            caches[f'A_{layer}'] = A
            

        # place or update final A, A L - 1 in dictionary of caches
        Z = self.linear(A, self.params[f'THETA_{L - 1}'], self.params[f'BETA_{L - 1}'])
        A = self.g(Z, self.ol_acts)
        caches[f'Z_{L}'] = Z
        caches[f'A_{L}'] = A

        return caches


    def backward(self, caches):
        L = self.L - 1

        # define dictionary of gradients
        grads = {}

        # calculate derivative of J with respect to 
        # last activation layer
        # if it is only A 2-layer network derive 
        # dJ_dAL, dAL_dZ and dZ_dTHETA/dZ_dBETA
        dJ_dAL = self.L_prime_A_last(caches)

        # access Z cache in last layer
        Z = caches[f'Z_{L}']
        A = caches[f'A_{L}']

        # calculate derivative of the activation of
        # the last layer with respect to Z
        dAL_dZ = self.g_prime(Z, self.ol_acts)

        # calculate error in last layer
        error = dJ_dAL * dAL_dZ

        # calculate gradients of coefficients in last layer
        grads[f'dZ_dTHETA_{L - 1}'] = (np.dot(error, caches[f'A_{L - 1}'].T) / self.num_instances) + ((self.lambda_ * self.params[f'THETA_{L - 1}']) / self.num_instances)
        grads[f'dZ_dBETA_{L - 1}'] = (np.sum(error, axis=1, keepdims=True) / self.num_instances)
        
        # if a 3-layer network access activations at last layer through caches
        # which goes through from 2 to 1 exclusively so... [2]
        for layer in range(L, 1, -1):
            Z = caches[f'Z_{layer - 1}']
            A = caches[f'A_{layer - 1}']

            # to access 2nd layer theta index will be 2 - 1
            THETA = self.params[f'THETA_{layer - 1}']

            # calculate derivative of activation 
            # in the layers preceding the last layer
            dA_dZ = self.g_prime(Z, self.hl_acts)

            # calculate error in the layers preceding the last layer
            error = np.dot(THETA.T, error) * dA_dZ

            # calculate gradients of coefficients preceding the last layer
            grads[f'dZ_dTHETA_{layer - 2}'] = (np.dot(error, caches[f'A_{layer - 2}'].T) / self.num_instances) + ((self.lambda_ * self.params[f'THETA_{layer - 2}']) / self.num_instances)
            grads[f'dZ_dBETA_{layer - 2}'] = (np.sum(error, axis=1, keepdims=True) / self.num_instances)

        return grads

    def linear(self, X, THETA, BETA):
        return np.dot(THETA, X) + BETA

    def g(self, Z, activation):
        if activation.lower() == "linear":
            return Z
        elif activation.lower() == "relu":
            A = np.maximum(0, Z)
        elif activation.lower() == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        elif activation.lower() == "tanh":
            A = np.tanh(Z)
        
        return A

    def g_prime(self, Z, activation):
        # if activation.lower()
        # what if activation is linear?
        # how to derive a linear activation
        if activation.lower() == "linear":
            return Z
        elif activation.lower() == "relu":
            # any of Zs values that are greater than 0 is true which
            # translates to 1 and 0 otherwise which is the derivative 
            # of a ReLU
            dA = (Z > 0).astype(int)
        elif activation.lower() == "sigmoid":
            dA = self.g(Z, "sigmoid") * (1 - self.g(Z, "sigmoid"))
        elif activation.lower() == "tanh":
            dA = 1 - np.power(self.g(Z, "tanh"), 2)

        return dA
    
    def update(self, grads):
        for layer in range(self.L - 1):
            CURR_THETA = copy.deepcopy(self.params[f'THETA_{layer}'])
            CURR_BETA = self.params[f'BETA_{layer}']

            self.params[f'THETA_{layer}'] = CURR_THETA - (self.learning_rate * grads[f'dZ_dTHETA_{layer}'])
            self.params[f'BETA_{layer}'] = CURR_BETA - (self.learning_rate * grads[f'dZ_dBETA_{layer}'])

    def accuracy(self, X_valid, Y_valid):
        m = X_valid.shape[1]
        # calculate y_pred and y_true here
        caches = self.forward(X_valid)

        # extract the predicted Y of the forward function
        # which is the node at the last layer
        Y_pred = caches[f'A_{self.L - 1}']

        # convert activations to either 0 or 1
        Y_pred = (Y_pred > 0.5).astype(int)
        # print(Y_pred)
        # print(Y_valid)

        # acc = accuracy_score(Y_valid, Y_pred)
        acc = np.sum((Y_pred == Y_valid) / m)
        return acc

def model_var_costs(costs_epochs):
    # [
    #     [(cost 1, cost 2, ...), (epoch 1, epoch 2, ...)],
    #     ...
    # ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    # place one inch padding between each subplot
    fig.tight_layout(pad=1)

    styles = [('p:', '#5d42f5'), ('h-', '#fc03a5'), ('o:', '#1e8beb'), ('x--','#1eeb8f'), ('+--', '#0eb802'), ('8-', '#f55600')]
    axes = axes.flat
    for index, cost_epoch in enumerate(costs_epochs):
        
        costs, epochs = cost_epoch[0], cost_epoch[1]

        axes[index].plot(costs, epochs, styles[index][0], c=styles[index][1], label='cost')
        axes[index].legend()
    
    plt.show()

def view_data_info(X_trains, Y_trains, X_cross, Y_cross, X_tests, Y_tests):
    print(f"X_trains: {X_trains.shape}")
    print(f"Y_trains: {Y_trains.shape}")
    print(f"X_cross: {X_cross.shape}")
    print(f"Y_cross: {Y_cross.shape}")
    print(f"X_tests: {X_tests.shape}")
    print(f"Y_tests: {Y_tests.shape}")

def load_dataset(name):
    # np.random.seed(2)
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    raw =   load_planar_dataset()
    # print(no_structure[0].shape)
    # print(no_structure[1].shape)
    datasets = {
        'planar': {
            'X': raw[0].T,
            'Y': raw[1].reshape(-1, 1)
        },
        'noisy_circles': {
            'X': noisy_circles[0],
            'Y': noisy_circles[1].reshape(noisy_circles[1].shape[0], 1)
        },
        'noisy_moons': {
            'X': noisy_moons[0],
            'Y': noisy_moons[1].reshape(noisy_circles[1].shape[0], 1)
        }, 
        'blobs': {
            'X': blobs[0],
            'Y': blobs[1].reshape(noisy_circles[1].shape[0], 1) % 2
        },
        'gaussian_quantiles': {
            'X': gaussian_quantiles[0],
            'Y': gaussian_quantiles[1].reshape(noisy_circles[1].shape[0], 1)
        },
        # 'no_structure': {
        #     'X': no_structure[0],
        #     'Y': no_structure[1].reshape(no_structure[1].shape[0], 1)
        # },
        # 'cats-and-dogs': {
        #     'X': 1
        # }
    }

    data = datasets[name]
    print(f"dataset shape: {data['X'].shape}")

    X_trains, _X, Y_trains, _Y = train_test_split(data['X'], data['Y'], test_size=0.3, random_state=0)
    X_tests, X_cross, Y_tests, Y_cross = train_test_split(_X, _Y, test_size=0.5, random_state=0)

    return X_trains, Y_trains, X_tests, Y_tests, X_cross, Y_cross 

if __name__ == "__main__":
    # Datasets
    dataset_name = sys.argv[1]
    
    # load dataset
    X_trains, Y_trains, X_tests, Y_tests, X_cross, Y_cross = load_dataset(dataset_name)

    # view data
    # view_data_info(X_trains, Y_trains, X_cross, Y_cross, X_tests, Y_tests)

    # hyper parameters
    lambdas = [10, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.125, 0.01]
    learning_rates = [1.2, 0.0075, 0.03, 0.01, 0.003, 0.001]
    architectures = [
        [2, 15, 15, 3, 1],
        [2, 10, 10, 10, 1],
        [2, 5, 5, 5, 1],
        [2, 30, 20, 10, 1],
        [2, 50, 50, 50, 25, 15, 10, 1]
    ]

    # models = []
    # model_var_costs_per_iter = []
    # for index, learning_rate in enumerate(learning_rates):
    model = NeuralNetwork(X_trains.T, Y_trains.T, epochs=50000, layers_dims=architectures[1], rec_ep_at=5000, learning_rate=learning_rates[2], regularization='l2', lambda_=lambdas[2])
    #     models.append(model)
    model.train()
    training_acc = model.accuracy(X_trains.T, Y_trains.T)
    cross_acc = model.accuracy(X_cross.T, Y_cross.T)
    test_acc = model.accuracy(X_tests.T, Y_tests.T)
    print('training accuracy: {:.2%}'.format(training_acc))
    print('cross accuracy: {:.2%}'.format(cross_acc))
    print('test accuracy: {:.2%}'.format(test_acc))
    #     model_var_costs_per_iter.append(model.costs_per_iter)
        
    # model_var_costs(model_var_costs_per_iter)



    

    
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from C1_W3.planar_utils import load_extra_datasets, load_planar_dataset, plot_decision_boundary

class ShallowNetwork:
    def __init__(self, data, epochs=10000, learning_rate=1.2, no_hidden_units=5, coeff_limit=0.01, split=True) -> None:
        
        if split is True:
            X_trains, X_tests, Y_trains, Y_tests = train_test_split(data['X'].T, data['Y'].T, test_size=0.3, random_state=0)
            self.X_trains = X_trains.T
            self.X_tests = X_tests.T
            self.Y_trains = Y_trains.T
            self.Y_tests = Y_tests.T
        else:
            self.X_trains, self.X_tests, self.Y_trains, self.Y_tests = data['X'], np.empty(0), data['Y'], np.empty(0)
            

        # n^[1st layer] x n^[0th layer]
        self._curr_theta_1 = np.random.randn(no_hidden_units, data['X'].shape[0]) * coeff_limit
        self._curr_beta_1 = np.random.randn(no_hidden_units, 1)

        # if y is m x n which has 2 dimensions then use n for 0th dimension 
        # of coefficient matrix else thenit means y is of 2 classes or continuous
        # values only meaning y vector is can be at most a 1d tensor or a vector

        # n^[2nd layer] x n^[1st layer]
        self._curr_theta_2 = np.random.randn(data['Y'].shape[0], no_hidden_units) * coeff_limit
        self._curr_beta_2 = np.random.randn(data['Y'].shape[0], 1)

        self.num_features = data['X'].shape[0]
        self.num_instances = data['X'].shape[1]
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.view_data()
        self.view_params()

    def view_data(self):
        # 2 x 400 and 1 x 400 for all X and Y respectively
        print('X_trains: {} \n'.format(self.X_trains.shape))
        print('X_trains shape: {} \n'.format(self.X_trains.shape))
        print('Y_trains: {} \n'.format(self.Y_trains))
        print('Y_trains shape: {} \n'.format(self.Y_trains.shape))

        print('X_tests: {} \n'.format(self.X_tests))
        print('X_tests shape: {} \n'.format(self.X_tests.shape))
        print('Y_tests: {} \n'.format(self.Y_tests))
        print('Y_tests shape: {} \n'.format(self.Y_tests.shape))

        
    def view_params(self):
        # 4 x 2 and 4 x 1 respectively
        print('theta_1: {} \n'.format(self._curr_theta_1))
        print('beta_1: {} \n'.format(self._curr_beta_1))

        # 1 x 4 and 1 x 1 respectively
        print('theta_2: {} \n'.format(self._curr_theta_2))
        print('beta_2: {} \n'.format(self._curr_beta_2))

    @property
    def params(self):
        return {
            'THETA_1': self._curr_theta_1,
            'BETA_1': self._curr_beta_1,
            'THETA_2': self._curr_theta_2,
            'BETA_2': self._curr_beta_2
        }
    
    @params.setter
    def params(self, kwargs):
        self._curr_theta_1 = kwargs['NEW_THETA_1']
        self._curr_beta_1 = kwargs['NEW_BETA_1']
        self._curr_theta_2 = kwargs['NEW_THETA_2']
        self._curr_beta_2 = kwargs['NEW_BETA_2']

    def J(self, A2):
        """
        A2 - is a predicted vector with 1 x m or (1 x 400)
        Y - is the output vector with 1 x m or (1 x 400)
        """

        cost = (np.dot(-1 * self.Y_trains, np.log(A2).T) - np.dot((1 - self.Y_trains), np.log(1 - A2).T)) / self.num_instances
        return cost

    def fit(self):
        # run algorithm for n epochs
        for epoch in range(self.epochs + 1):
            Y_pred, cache = self.forward(self.X_trains)
            if epoch % 1000 == 0:
                print('epoch {} \n'.format(epoch))
                print('current cost: {} \n'.format(self.J(Y_pred)))
            self.optimize(cache)

        print('DONE')

    def optimize(self, cache):
        grads = self.backward(cache)
        print('current gradients: {} \n'.format(grads))

        THETA_2 = copy.deepcopy(self.params['THETA_2'])
        BETA_2 = self.params['BETA_2']
        THETA_1 = copy.deepcopy(self.params['THETA_1'])
        BETA_1 = self.params['BETA_1']

        new_theta_2 = THETA_2 - (self.learning_rate * grads['dTHETA_2'])
        new_beta_2 = BETA_2 - (self.learning_rate * grads['dBETA_2'])
        new_theta_1 = THETA_1 - (self.learning_rate * grads['dTHETA_1'])
        new_beta_1 = BETA_1 - (self.learning_rate * grads['dBETA_1'])

        self.params = {
            'NEW_THETA_1': new_theta_1,
            'NEW_BETA_1': new_beta_1,
            'NEW_THETA_2': new_theta_2,
            'NEW_BETA_2': new_beta_2,
        }

    def linear(self, curr_theta, X, curr_beta):
        return np.dot(curr_theta, X) + curr_beta

    def sigmoid_prime(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def tanh_prime(self, Z):
        return 1 - np.power(self.tanh(Z), 2)

    def tanh(self, Z):
        return np.tanh(Z)
    
    def forward(self, X):
        """
        X - is an n x m matrix, in this case 2 x 400
        Y - is an k x m column vector where k is the number of 
        classes, in this case 1 x 400
        """

        # 4 x 2 and 4 x 1 respectively
        THETA_1 = self.params['THETA_1']
        BETA_1 = self.params['BETA_1']

        # 4 x 1 and 1 x 1 respectively
        THETA_2 = self.params['THETA_2']
        BETA_2 = self.params['BETA_2']

        # results in 400 x 4 matrix
        Z1 = self.linear(THETA_1, X, BETA_1)
        A1 = self.tanh(Z1)

        # results in 400/400 x 1 matrix
        Z2 = self.linear(THETA_2, A1, BETA_2)
        A2 = self.sigmoid(Z2)

        cache = {
            'A1': A1,
            'A2': A2
        }

        Y_pred = A2

        return Y_pred, cache
    
    def backward(self, cache):
        X, Y = self.X_trains, self.Y_trains

        A1, A2 = cache['A1'], cache['A2']
        THETA_1, THETA_2 = self.params['THETA_1'], self.params['THETA_2']

        ERROR_L2 = A2 - Y
        dTHETA_2 = np.dot(ERROR_L2, A1.T) / self.num_instances
        dBETA_2 = np.sum(ERROR_L2, axis=1, keepdims=True) / self.num_instances

        ERROR_L1 = np.dot(THETA_2.T, ERROR_L2) * self.tanh_prime(A1)
        dTHETA_1 = np.dot(ERROR_L1, X.T) / self.num_instances
        dBETA_1 = np.sum(ERROR_L1, axis=1, keepdims=True) / self.num_instances

        grads = {
            'dTHETA_2': dTHETA_2,
            'dBETA_2': dBETA_2,
            'dTHETA_1': dTHETA_1,
            'dBETA_1': dBETA_1
        }

        return grads
    
    def compare():
        # try out different hyperparameters
        # - epoch
        # - learning rate
        # - no. of nodes in 2nd layer
        pass

    def plot_cost_per_epoch():
        pass

    def plot_decision_boundary():
        pass

    def predict():
        pass

def view_data(data):
    # Visualize the data
    plt.scatter(data['X'][0, :], data['X'][1, :], c=data['Y'], s=40, cmap=plt.cm.Spectral)
    plt.show()

    print(data['X'])
    print(data['X'].shape)
    print(data['Y'])
    print(data['Y'].shape)



if __name__ == "__main__":
    # Datasets
    dataset_name = sys.argv[1]

    # np.random.seed(2)
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    raw =   load_planar_dataset()
    datasets = {
        'planar': {
            'X': raw[0],
            'Y': raw[1]
        },
        'noisy_circles': {
            'X': noisy_circles[0].T,
            'Y': noisy_circles[1].reshape(1, noisy_circles[1].shape[0])
        },
        'noisy_moons': {
            'X': noisy_moons[0].T,
            'Y': noisy_moons[1].reshape(1, noisy_circles[1].shape[0])
        }, 
        'blobs': {
            'X': noisy_moons[0].T,
            'Y': noisy_moons[1].reshape(1, noisy_circles[1].shape[0]) % 2
        },
        'gaussian_quantiles': {
            'X': noisy_moons[0].T,
            'Y': noisy_moons[1].reshape(1, noisy_circles[1].shape[0])
        },
        'no_structure': {
            'X': noisy_moons[0].T,
            'Y': noisy_moons[1].reshape(1, noisy_circles[1].shape[0])
        }
    }

    data = datasets[dataset_name]
    # view_data(data)
    
    # Shallow Network model
    model = ShallowNetwork(data, split=True)
    model.fit()

    # Logistic Regression model
    
    

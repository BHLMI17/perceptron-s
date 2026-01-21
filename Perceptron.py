import numpy as np

class Perceptron(object):
    
    #eta: learning rate (between 0.0 and 1.0) -> defines how quickly the model learns
    #n_iter: number of iterations that passes over the training dataset
    #w_ : weights of the the input values


    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter


    def weighted_sum(self, X):
        # Calculate the weighted sum of inputs
        # Uses the dot product between X and weights
        return np.dot(X, self.w_[1:]) + self.w_[0]
    

    def predict(self, X):
        # Return class label after unit step
        #unit step function activation
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
    

    def fit(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):

                y_pred = self.predict(xi)
                update = self.eta * (target - y_pred)

                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(update != 0)

            self.errors_.append(errors)

        return self
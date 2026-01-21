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

        #initializing weights to 0
        self.w_=np.zeros(1+X.shape[1])
        self.errors_=[]

        print("Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0

            #loop through each input
            for xi, y in zip(X, y):

                #1- calculate the ypred (predicted value)
                y_pred = self.predict(xi)

                #2- calculate update
                #update = n * (y-ypred)
                update = self.eta * (y - y_pred)
                
                #3- update the weights
                #(new)Wi = (old)Wi + (change)Wi where change(Wi) = n * (y - ypred) = update * Xi
                self.w_[1:] = self.w_[1:] + update *xi
                print("Updated Weights:", self.w_[1:])

                #update the bias(Xo = 1)
                self.w_[0] = self.w_[0] + update

                #if update != 0, it means that ypred != y
                error += int(update != 0.0)
                self.errors_.append(error)
        return self

import numpy as np
import pandas as pd

def train_model(X, y, alpha, max_epoch):
    '''
    X: input features
    y: responses
    alpha: learning rate
    max_epoch: maximum epochs

    returns w, hist_loss
    '''
    hist_loss = []

    np.random.seed(42)
    w = np.random.rand(5,)

    for _ in range(max_epoch):
        # make prediction
        yhat = predict(w, X)

        # compute the gradient
        #gradient = compute_gradient()

        # update loss value history
        loss = loss_fn(y, yhat)
        hist_loss.append(loss)
        print(f'iter {_ + 1}: loss = {loss}')

        # update w
        #w = w - (alpha * gradient)

    return w, hist_loss

def predict(w, X):
    '''
    w: weights
    X: input features
    '''
    return np.dot(X, w.T)

def loss_fn(y, yhat):
    '''
    y: responses
    yhat: predicted values
    '''
    # for regression we will use MSE
    # MSE = 1/2n * sum_of_squared_errors
    # 1/2 for convenience when we find its derivative
    # should be fine as we don't change the minimum when we multiply the loss_fn by a scalar
    return sum([(y_i - yhat_i) ** 2 for y_i, yhat_i in zip(y, yhat)] )/ (2 * len(y))

'''
1. display training loss for each epoch
2. display estimated weights (after model training)
3. dipslay training loss against epoch graph
4. evaluate linear regression model with estimated weights on the test 
    set and display at least R squared, MSE and MAE
'''
df = pd.read_csv('assignment1_dataset.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(X.shape)
print(y.shape)

estimated_w, hist_loss = train_model(X, y, 0.1, 300)

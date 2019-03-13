import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    
    #first step distract max in order not to overflow float - see http://cs231n.github.io/linear-classify/#softmax Practical Issues
    res_predictions = predictions.copy()
    res_predictions -= np.max(predictions,axis = 1)[:, None]
    return np.exp(res_predictions) / np.sum(np.exp(res_predictions), axis = 1) [:, None]
#     res_predictions -= np.max(predictions)
#     return np.exp(res_predictions) / np.sum(np.exp(res_predictions)) 


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (N, batch_size) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    return -np.mean(np.log(probs[:, target_index]))
#     return -np.log(probs[target_index])


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    
    loss = cross_entropy_loss(softmax(predictions), target_index)
    dprediction =  softmax(predictions)
    
#     dprediction[target_index] -= 1
    dprediction[: ,target_index] -= 1
    dprediction = dprediction / target_index.shape[0]
   
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(np.square(W))    
    grad = 2 * reg_strength * W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W) #num_batch * classes
    
    loss, dpredictions = softmax_with_cross_entropy(predictions.T, target_index) #classes * num_batch

    dW = np.dot(dpredictions, X).T 
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            for batch_indices in batches_indices:
                cur_train_X = X[batch_indices]
                cur_train_y = y[batch_indices]
                loss, dW = linear_softmax(cur_train_X, self.W, cur_train_y)
                loss_reg, dW_reg = l2_regularization(self.W, reg)
                loss += loss_reg
                dW += dW_reg
                self.W = self.W - learning_rate * dW

            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.dot(X, self.W)
        

        return np.argmax(y_pred, axis = 1)



                
                                                          

            

                

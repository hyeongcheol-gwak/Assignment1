################################### Problem 1.1 ###################################

def learn_mul(X, y):
    ################# YOUR CODE COMES HERE ######################
    # training and return the multi-class logistic model
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    #############################################################
    return lr

def inference_mul(x, lr_model):
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values
    y_pred = lr_model.predict(x)
    #############################################################
    return y_pred


################################### Problem 1.2 ###################################


def learn_ovr(X, y):
    lrs = []
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
    for i in range(num_classes):
        print('training %s classifier'%(ordinal(i+1)))
        ################# YOUR CODE COMES HERE ######################
        # training and return the multi-class logistic model
        from sklearn.linear_model import LogisticRegression
        y_binary = (y == i).astype(int)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y_binary)
        lrs.append(lr)
        #############################################################

    return lrs

def inference_ovr(X, lrs):
    ################# YOUR CODE COMES HERE ######################
    import numpy as np
    # 1D 단일 샘플이 들어오더라도 2D 배열인 (1, n) 형태로 안전하게 변환
    X_2d = np.atleast_2d(X) 
    
    probas = []
    for lr in lrs:
        probas.append(lr.predict_proba(X_2d)[:, 1])

    y_pred = np.argmax(probas, axis=0)
    
    if np.ndim(X) == 1:
        return y_pred[0]
    #############################################################
    return y_pred

################################### Problem 1.3 ###################################


def learn_ovo(X, y):
    lrs = {}
    class_pairs = list(combinations(range(num_classes), 2))
    for i, j in class_pairs:
        print(f'training classifier for class {i} vs {j}')
        ################# YOUR CODE COMES HERE ######################
        # training and return the multi-class logistic model
        from sklearn.linear_model import LogisticRegression
        mask = (y == i) | (y == j)
        X_sub, y_sub = X[mask], y[mask]
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_sub, y_sub)
        lrs[(i, j)] = lr
        #############################################################

    return lrs

def inference_ovo(X, lrs):
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values
    import numpy as np
    num_classes = max([max(pair) for pair in lrs.keys()]) + 1
    votes = np.zeros((X.shape[0], num_classes))
    for (_, _), lr in lrs.items():
        preds = lr.predict(X).astype(int)
        np.add.at(votes, (np.arange(X.shape[0]), preds), 1)
    y_pred = np.argmax(votes, axis=1)
    #############################################################
    return y_pred


################################### Problem 2   ###################################


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000): # <<< You can add your own input parameters
        ################# YOUR CODE COMES HERE ######################
        # initialize class member variable
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        #############################################################

    def sigmoid(self, z):
        # YOUR CODE COMES HERE
        import numpy as np
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        ################# YOUR CODE COMES HERE ######################
        # training model here
        import numpy as np
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        #############################################################
        return

    def predict(self, X):
        ################# YOUR CODE COMES HERE ######################
        # return predicted y
        import numpy as np
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = np.where(y_predicted > 0.5, 1, 0)
        return y_predicted_cls
        #############################################################
        return

    # You can add your own member functions
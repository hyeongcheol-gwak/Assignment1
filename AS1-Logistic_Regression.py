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
    # inference model and return predicted y values
    import numpy as np
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
    X_2d = np.atleast_2d(X)
    num_classes = max([max(pair) for pair in lrs.keys()]) + 1
    votes = np.zeros((X_2d.shape[0], num_classes))
    for (_, _), lr in lrs.items():
        preds = lr.predict(X_2d).astype(int)
        np.add.at(votes, (np.arange(X_2d.shape[0]), preds), 1)
    y_pred = np.argmax(votes, axis=1)
    if np.ndim(X) == 1:
        return y_pred[0]
     #############################################################
    return y_pred


################################### Problem 2   ###################################


class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=2000, lambda_param=1.0, poly_degree=2): # <<< You can add your own input parameters
        ################# YOUR CODE COMES HERE ######################
        # initialize class member variable
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.poly_degree = poly_degree
        self.weights = None
        self.bias = None
        #############################################################

    def sigmoid(self, z):
        # YOUR CODE COMES HERE
        import numpy as np
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        ################# YOUR CODE COMES HERE ######################
        # training model here
        import numpy as np
        def add_poly_features(X_in):
            if self.poly_degree < 2:
                return X_in
            features = [X_in]
            n = X_in.shape[1]
            for i in range(n):
                for j in range(i, n):
                    features.append((X_in[:, i] * X_in[:, j]).reshape(-1, 1))
            return np.hstack(features)
        X = add_poly_features(X)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        # Adam optimizer state
        m_w = np.zeros(num_features)
        v_w = np.zeros(num_features)
        m_b = 0.0
        v_b = 0.0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for t in range(1, self.num_iterations + 1):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y)) + (self.lambda_param / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(y_predicted - y)
            # Adam moment updates
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
            # Bias-corrected estimates
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        #############################################################
        return

    def predict(self, X):
        ################# YOUR CODE COMES HERE ######################
        # return predicted y
        import numpy as np

        def add_poly_features(X_in):
            if self.poly_degree < 2:
                return X_in
            features = [X_in]
            n = X_in.shape[1]
            for i in range(n):
                for j in range(i, n):
                    features.append((X_in[:, i] * X_in[:, j]).reshape(-1, 1))
            return np.hstack(features)

        X = add_poly_features(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = np.where(y_predicted > 0.5, 1, 0)
        #############################################################
        return y_predicted_cls

    # You can add your own member functions
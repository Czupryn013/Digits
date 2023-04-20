import numpy as np
import pandas as pd


def unit_step(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, bias=0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step
        self.weights = None
        self.bias = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randint(2, size=n_features)
        self.weights = self.weights.astype(float)
        y_ = np.where(y > 0, 1, 0) #czemu "_"

        #learn
        for _ in range(self.n_iters):
            for index, x_i in enumerate(X): #enumerate ?
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                #update
                update = self.lr  * (y_[index] - y_predicted) #int - np_array?
                print(y_[index].shape)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


# Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    X = pd.read_csv("./data/train.csv")
    y = X["Survived"]
    y =  np.where(y, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000, bias=0.1)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()
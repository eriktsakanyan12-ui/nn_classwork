import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Perceptron:
  def __init__(self, num_inputs, learning_rate=0.003):
    self.weights = np.random.rand(num_inputs+1)
    self.learning_rate = learning_rate

  def weighted_sum(self, inputs):
    Z = np.dot(inputs, self.weights[1:]) + self.weights[0]
    return Z

  def predict(self, x):
    z = np.dot(x, self.weights[1:])+self.weights[0]
    return z
    
  def loss(self, prediction, target):
    return np.mean((prediction-target)**2)
  
  def stoch(self, X, y, tolerance=10e-5, n_epochs=100):
     return self.fit(X, y, tolerance, n_epochs, batch_size = 1)

  def batch(self, X, y, tolerance=10e-5, n_epochs=100):
     return self.fit(X, y, tolerance, n_epochs, batch_size = 1000)

  def mini_batch(self, X, y, tolerance=10e-5, n_epochs=100):
     return self.fit(X, y, tolerance, n_epochs, batch_size = 10)
  
  def fit(self, X, y, tolerance=10e-5, n_epochs=100, batch_size=1000):
    history = {'k': [], 'b': [], 'mse': []}
    n_samples = len(X)

    for epoch in range(n_epochs):
        lr = self.learning_rate * (0.95 ** epoch)

        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_pred = self.predict(X_batch) #forward pass

            error = y_batch - y_pred
            mse = np.mean(error**2)

            if mse < tolerance:
                return history
            
            #backward pass
            grad_w = -2 * np.dot(X_batch.T, error) / batch_size
            grad_b = -2 * np.mean(error)
            self.weights[1:] -= lr * grad_w
            self.weights[0]  -= lr * grad_b

            history['mse'].append(mse)
            history['k'].append(self.weights[1])
            history['b'].append(self.weights[0])

    return history 


if __name__=="__main__":

    # from sklearn.datasets import make_regression
    
    # X, y = make_regression(
    #     n_samples=1000,
    #     n_features=1,
    #     n_targets=1,
    #     random_state=42
    # )
    # nn = Perceptron(1)
    # nn.fit(X, y)
    # pred = nn.predict(X)
    # print(nn.loss(pred, y))

    k = 5
    b = 3
    X = np.linspace(-10, 10, 1000)
    y = k * X + b
    error = np.random.normal(0, 2, size = X.shape[0])
    # error = np.linspace(0, 2, 1000)
    # np.random.shuffle(error)
    y_synt = y + error
    # plt.plot(X, y_synt, "o", c = "r")
    # plt.plot(X, y)

    nn = Perceptron(1)
    h = nn.batch(X.reshape(-1, 1), y_synt)
    final_pred = nn.predict(X.reshape(-1, 1))
    # plt.plot(h['mse'], h['k'], "o")
    # plt.plot(h['mse'])
    # plt.show()
    
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
       x = h['k'],
       y = h['b'],
       z = h['mse'],
       mode = 'markers',
       marker = dict(
          size = 5,
          color = h['mse'],
       )
    ))

    fig.show()

    print(f"K: {nn.weights[1].item():.3f}")
    print(f"B: {nn.weights[0]:.3f}")
    print(f"MSE: {nn.loss(final_pred, y_synt):.3f}")
    
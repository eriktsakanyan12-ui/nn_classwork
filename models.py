import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Perceptron:
  def __init__(self, num_inputs, learning_rate=0.01):
    self.weights = np.random.randn(num_inputs + 1)
    self.learning_rate = learning_rate

  def weighted_sum(self, inputs):
    Z=np.dot(inputs, self.weights[1:]) + self.weights[0]
    return Z
  
  def predict(self, X):
    z=self.weighted_sum(X)
    return z
  
  def loss(self, prediction, target):
    return prediction - target 
  
  def stoch(self, X, y, tolerance=1e-5, n_epochs=1000):
    return self.fit(X, y, tolerance, n_epochs, batch_size=1)
  def batch(self, X, y, tolerance=1e-5, n_epochs=1000):
    return self.fit(X, y, tolerance, n_epochs, batch_size=1000)
  def mini_batch(self, X, y, batch_size=32, tolerance=1e-5, n_epochs=1000):
    return self.fit(X, y, tolerance, n_epochs, batch_size=10)

  def fit(self, X, y, tolerance=1e-5, n_epochs=100, batch_size=1000):
    self.history = []        # store MSE
    self.weight_history = [] # store weights [k,b]
    n_samples = len(X)

    for epoch in range(n_epochs):
        lr = self.learning_rate * (0.95 ** epoch)
        indices = np.random.permutation(n_samples)
        X_sh = X[indices]
        y_sh = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_sh[i:i+batch_size]
            y_batch = y_sh[i:i+batch_size]

            y_pred = self.predict(X_batch)
            error = y_batch - y_pred
            mse = np.mean(error**2)

            if mse < tolerance:
                return

            # gradient update
            grad_w = -2 * np.dot(X_batch.T, error) / len(X_batch)
            grad_b = -2 * np.mean(error)
            self.weights[1:] -= lr * grad_w
            self.weights[0] -= lr * grad_b

            # save history
            self.history.append(mse)
            self.weight_history.append(self.weights.copy())


if __name__ == "__main__":
   k, b = 1, 2

   x = np.linspace(-10, 11, 1000)
   x = x.reshape(-1, 1)

   y = k*x + b
   y = y.flatten()

   random_int = (np.random.rand(1000) - 0.5) * 10
   y_synt = y + random_int

   nn = Perceptron(1, learning_rate=0.001)
   nn.fit(x, y_synt)

   plt.plot(x, y_synt, 'o', c = 'r')
   plt.plot(x, y)
   plt.show()
   print("Underlying clean line values (first 10):", y[:10])

   nn_mb = Perceptron(1, learning_rate=0.001)
   nn_mb.fit(x, y_synt, n_epochs=50, batch_size=32)
   print("Last 5 mse values (mini-batch):", nn_mb.history[-5:])

   k_vals = [w[1] for w in nn_mb.weight_history]
   b_vals = [w[0] for w in nn_mb.weight_history]
   mse_vals = nn_mb.history
   fig = go.Figure()
   fig.add_trace(go.Scatter3d(
      x=k_vals,
      y=b_vals,
      z=mse_vals,
      mode='markers',
      marker=dict(
          size=4,
          color=mse_vals,
          colorscale='Viridis'
      )
  ))
   fig.update_layout(
      scene=dict(
          xaxis_title="k (slope)",
          yaxis_title="b (bias)",
          zaxis_title="MSE"
      )
  )
   fig.show()

   nn = Perceptron(1)
   nn.fit(x, y)
   print(nn.predict(x)[:5])
   print("Weights:", nn.weights)

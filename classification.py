import numpy as np
from sklearn.datasets import make_blobs
import plotly.graph_objects as go

class simple_nn:
  def __init__(self, num_inputs, learning_rate=0.003, hidden_size=1):
    self.lr = learning_rate
    self.W1 = np.random.randn(num_inputs, hidden_size)
    self.b1 = np.zeros((1, hidden_size))
    self.W2 = np.random.randn(hidden_size, 1)
    self.b2 = np.zeros((1, 1))

  def forward_propagation(self, X, W1, W2, b1, b2):
    z1 = np.dot(X, W1) + b1
    a1 = self.relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = self.sigmoid(z2)

    return z1, a1, z2, a2

  def relu(self, x):
    return np.maximum(0, x)
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def backward_propagation(self, X, y, z1, a1, z2, a2, W1, W2, b1, b2):
    grad_L_z2 = a2 - y
    grad_L_W2 = np.dot(a1.T, grad_L_z2)
    grad_L_b2 = np.sum(grad_L_z2, axis = 0, keepdims=True)

    grad_L_a1 = np.dot(grad_L_z2, W2.T)
    grad_L_z1 = grad_L_a1 * self.relu_derivative(z1)
    grad_L_W1 = np.dot(X.T, grad_L_z1)
    grad_L_b1 = np.sum(grad_L_z1, axis = 0, keepdims=True)

    return grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2
  
  def relu_derivative(self, x):
    return np.where(x > 0, 1, 0)
  
  def update_parameters(self, W1, W2, b1, b2, grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2):
    W1 = W1 - self.lr * grad_L_W1
    W2 = W2 - self.lr * grad_L_W2
    b1 = b1 - self.lr * grad_L_b1
    b2 = b2 - self.lr * grad_L_b2

    return W1, W2, b1, b2
  
  def predict(self, X, threshold = 0.5):
     z1, a1, z2, a2 = self.forward_propagation(X, self.W1, self.W2, self.b1, self.b2)
     return a2 >= threshold
  
  def compute_loss(self, y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
  
  # def train(self, X, y, W1, W2, b1, b2, epochs, ):
  #   W1_history, W2_history, b1_history, b2_history, loss_history = [], [], [], [], []

  #   for epoch in range(epochs):
  #     z1, a1, z2, a2 = self.forward_propagation(X, W1, W2, b1, b2)

  #     loss = self.compute_loss(y, a2)
  #     loss_history.append(loss)
      
  #     W1_history.append(W1.copy())
  #     W2_history.append(W2.copy())
  #     b1_history.append(b1.copy())
  #     b2_history.append(b2.copy())

  #     grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2 = self.backward_propagation(X, y, z1, a1, z2, a2, W1, W2, b1, b2)

  #     W1, W2, b1, b2 = self.update_parameters(W1, W2, b1, b2, grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2)

  #     if epoch % 100 == 0:
  #       print(f'Epoch {epoch}, Loss: {loss}')

  #   return W1, W2, b1, b2, W1_history, W2_history, b1_history, b2_history, loss_history
  
  def train(self, X, y, epochs=500, tolerance=10e-5, batch_size=None):
    history = {'W1': [], 'W2': [], 'b1': [], 'b2': [], 'loss': []}
    n_samples = X.shape[0]

    if batch_size is None:
        batch_size = n_samples

    for epoch in range(epochs):
        self.lr *= 0.95 ** epoch

        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            z1, a1, z2, a2 = self.forward_propagation(X_batch, self.W1, self.W2, self.b1, self.b2)

            loss = self.compute_loss(y_batch, a2)
            history['loss'].append(loss)

            if loss < tolerance:
              return history
            
            grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2 = self.backward_propagation(
                X_batch, y_batch, z1, a1, z2, a2, self.W1, self.W2, self.b1, self.b2
            )

            self.W1, self.W2, self.b1, self.b2 = self.update_parameters(
                self.W1, self.W2, self.b1, self.b2,
                grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2
            )

            history['W1'].append(self.W1.copy())
            history['W2'].append(self.W2.copy())
            history['b1'].append(self.b1.copy())
            history['b2'].append(self.b2.copy())

        if epoch % 2 == 0:
          print(f'Epoch {epoch}, Loss: {loss}')

    return history
  

if __name__=="__main__":

  X, y = make_blobs(n_samples=500, centers=2, n_features=1, random_state=42)
  y = y.reshape(-1,1)   

  input_size = X.shape[1]
  hidden_size = 1

  W1 = np.random.randn(input_size, hidden_size)
  b1 = np.zeros((1, hidden_size))
  W2 = np.random.randn(hidden_size,1)
  b2 = np.zeros((1,1))


  nn = simple_nn(num_inputs=input_size)
  history = nn.train(X, y, epochs=50)
  y_pred = nn.predict(X)
  accuracy = np.mean(y_pred == y)
  print(accuracy)

  w1_values = [w[0,0] for w in history['W1']]
  b1_values = [b[0,0] for b in history['b1']]
  loss_values = history['loss'][:len(w1_values)]
  # print(len(w1_values), len(b1_values), len(loss_values))

  fig = go.Figure()
  fig.add_trace(go.Scatter3d(
      x=w1_values,
      y=b1_values,
      z=loss_values,
      mode='markers',
      marker=dict(
          size=5,
          color=loss_values,
          colorscale='Viridis'
      )
  ))
  fig.show()

import numpy as np

class Perceptron:
  def __init__(self, num_inputs, learning_rate=0.01):
    self.weights = np.random.randn(num_inputs + 1)
    self.lr = learning_rate

  def forward_propogation(self, X, W1, W2, b1, b2):
    z1=np.dot(X, W1) + b1
    a1=self.relu(z1)

    z2=np.dot(a1, W2) + b2
    a2=self.sigmoid(z2)
    return z1, a1, z2, a2
  
  def relu(self, x):
    return np.maximum(0, x)
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def backward_propogation(self, X, y, z1, a1, z2, a2, W1, W2, b1, b2):
    grad_L_z2 = a2-y
    grad_L_W2 = np.dot(a1.T, grad_L_z2)
    grad_L_b2 = np.sum(grad_L_z2, axis=0, keepdims=True)

    grad_L_a1 = np.dot(grad_L_z2, W2.T)
    grad_L_z1 = grad_L_a1 * self.relu_derivative(z1)
    grad_L_W1 = np.dot(X.T, grad_L_z1)
    grad_L_b1 = np.sum(grad_L_z1, axis=0, keepdims=True)

    return grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2
  
  def relu_derivative(self, x):
    return np.where(x > 0, 1, 0)
  
  def update_parameters(self, W1, W2, b1, b2, grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2, lr):
    W1 = W1 - self.lr * grad_L_W1
    W2 = W2 - self.lr * grad_L_W2
    b1 = b1 - self.lr * grad_L_b1
    b2 = b2 - self.lr * grad_L_b2
    return W1, W2, b1, b2
  
  def compute_loss(self, y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
  
  def train(self, X, y, W1, W2, b1, b2, lr, epochs=100):
    W1_history, W2_history, b1_history, b2_history, loss_history = [], [], [], [], []

    for epoch in range(epochs): 
      z1, a1, z2, a2 = self.forward_propogation(X, W1, W2, b1, b2)
        
      loss = self.compute_loss(y, a2)
      loss_history.append(loss)

      W1_history.append(W1.copy())
      W2_history.append(W2.copy())
      b1_history.append(b1.copy())
      b2_history.append(b2.copy())
      
      grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2 = self.backward_propogation(X, y, z1, a1, z2, a2, W1, W2, b1, b2)
      W1, W2, b1, b2 = self.update_parameters(W1, W2, b1, b2, grad_L_W1, grad_L_W2, grad_L_b1, grad_L_b2, lr)

      if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, W2, b1, b2, W1_history, W2_history, b1_history, b2_history, loss_history
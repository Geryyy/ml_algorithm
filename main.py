import numpy as np

def loss(y_ls, x_ls, w, b):
    wx = np.dot(w, x_ls)
    loss_values = np.log(1 + np.exp(-y_ls * (wx + b)))
    return np.sum(loss_values)

def grad_loss(y_ls, x_ls, w, b):
    wx = np.dot(w, x_ls)
    exp_term = np.exp(-y_ls * (wx + b))
    
    grad_term = exp_term / (1 + exp_term)
    grad = -y_ls * np.vstack([x_ls, np.ones((1, x_ls.shape[1]))]) * grad_term
    print("shape grad: ", grad.shape)
    
    # Sum the gradients over all samples
    return np.sum(grad, axis=1, keepdims=True)

# Example usage
N = 5
x_ls = np.random.randn(2, N)
y_ls = np.array([1, -1, 1, -1, 1])
w = np.array([0.5, 0.5])
b = 0.1

loss_value = loss(y_ls, x_ls, w, b)
grad_value = grad_loss(y_ls, x_ls, w, b)

print("Loss:", loss_value)
print("Gradient:", grad_value)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate data (x and y)
N = 100  # Number of data points
x = np.random.randn(2, N)
y = np.random.choice([-1, 1], N)

# Fix the value of b
b = 0.5

# Create a meshgrid for w1 and w2
w1_values = np.linspace(-2, 2, 100)
w2_values = np.linspace(-2, 2, 100)
w1_mesh, w2_mesh = np.meshgrid(w1_values, w2_values)
loss_values = np.zeros_like(w1_mesh)

# Calculate loss for each combination of w1 and w2
for i in range(len(w1_values)):
    for j in range(len(w2_values)):
        w = np.array([w1_values[i], w2_values[j]])
        loss_values[i, j] = loss(y, x, w, b)

# Create a 3D surface plot for the loss

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plot the loss as a surface
surface = ax.plot_surface(w1_mesh, w2_mesh, loss_values, cmap='viridis')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('Loss as a Surface')

# Calculate the gradient for the loss at the minimum point
w_min = np.unravel_index(loss_values.argmin(), loss_values.shape)
w_opt = np.array([w1_values[w_min[0]], w2_values[w_min[1]]])
grad = grad_loss(y, x, w_opt, b)

# Plot the gradient as a red sphere at the minimum point
ax.scatter(w_opt[0], w_opt[1], loss(y, x, w_opt, b), c='red', s=30, marker='o')


plt.show()
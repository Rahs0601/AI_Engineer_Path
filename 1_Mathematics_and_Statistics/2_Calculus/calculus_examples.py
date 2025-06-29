# Calculus Examples (Numerical Approximations and Gradient Descent)

import numpy as np

# 1. Numerical Differentiation (Approximating Derivatives)
print("--- 1. Numerical Differentiation ---")

def f(x):
    return x**2

def numerical_derivative(func, x, h=0.0001):
    # Using the central difference formula for better accuracy
    return (func(x + h) - func(x - h)) / (2 * h)

x_val = 3
derivative_at_x = numerical_derivative(f, x_val)
print(f"Function f(x) = x^2")
print(f"Numerical derivative of f(x) at x = {x_val}: {derivative_at_x}")
print(f"Analytical derivative (2x) at x = {x_val}: {2 * x_val}")

# 2. Partial Derivatives (Numerical Approximation)
print("--- 2. Partial Derivatives ---")

def g(x, y):
    return x**2 + y**3

def numerical_partial_derivative_x(func, x, y, h=0.0001):
    return (func(x + h, y) - func(x - h, y)) / (2 * h)

def numerical_partial_derivative_y(func, x, y, h=0.0001):
    return (func(x, y + h) - func(x, y - h)) / (2 * h)

x_val_g = 2
y_val_g = 3

partial_x = numerical_partial_derivative_x(g, x_val_g, y_val_g)
partial_y = numerical_partial_derivative_y(g, x_val_g, y_val_g)

print(f"Function g(x, y) = x^2 + y^3")
print(f"Numerical partial derivative ∂g/∂x at ({x_val_g}, {y_val_g}): {partial_x}")
print(f"Analytical partial derivative ∂g/∂x (2x) at ({x_val_g}, {y_val_g}): {2 * x_val_g}")
print(f"Numerical partial derivative ∂g/∂y at ({x_val_g}, {y_val_g}): {partial_y}")
print(f"Analytical partial derivative ∂g/∂y (3y^2) at ({x_val_g}, {y_val_g}): {3 * (y_val_g**2)}")

# 3. Gradient (Numerical Approximation)
print("--- 3. Gradient ---")

def numerical_gradient(func, point, h=0.0001):
    grad = np.zeros_like(point, dtype=float)
    for i in range(len(point)):
        temp_plus = np.copy(point)
        temp_minus = np.copy(point)
        temp_plus[i] += h
        temp_minus[i] -= h
        grad[i] = (func(*temp_plus) - func(*temp_minus)) / (2 * h)
    return grad

# For the function g(x, y) = x^2 + y^3
point_g = np.array([x_val_g, y_val_g])
gradient_g = numerical_gradient(g, point_g)
print(f"Numerical gradient of g(x, y) at {point_g}: {gradient_g}")
print(f"Analytical gradient of g(x, y) at {point_g}: [{2 * x_val_g}, {3 * (y_val_g**2)}]")

# 4. Gradient Descent Example
print("--- 4. Gradient Descent Example ---")

# Define a simple cost function: f(x) = (x - 5)^2
def cost_function(x):
    return (x - 5)**2

# Define its derivative: f'(x) = 2 * (x - 5)
def derivative_cost_function(x):
    return 2 * (x - 5)

# Gradient Descent parameters
learning_rate = 0.1
initial_x = 0.0
iterations = 50

x_history = [initial_x]

x = initial_x
for i in range(iterations):
    gradient = derivative_cost_function(x)
    x = x - learning_rate * gradient
    x_history.append(x)

print(f"Cost function: f(x) = (x - 5)^2")
print(f"Initial x: {initial_x}")
print(f"Learning Rate: {learning_rate}")
print(f"Iterations: {iterations}")
print(f"Final x after gradient descent: {x:.4f}")
print(f"Minimum of f(x) is at x = 5.0")

# You can plot x_history to visualize the convergence
# import matplotlib.pyplot as plt
# plt.plot(x_history, [cost_function(val) for val in x_history], marker='o')
# plt.title('Gradient Descent Progress')
# plt.xlabel('x value')
# plt.ylabel('Cost Function Value')
# plt.grid(True)
# plt.show()

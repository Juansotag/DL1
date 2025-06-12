import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

def optimize_torch_fun1(f):
    """
    Minimizes a PyTorch-based function f(x), where x is a tensor of shape (2,).
    Returns the optimized tensor x.
    """
    tensor = torch.rand(2, requires_grad=True)  # Initial vector of shape (2,)
    learning_rate, num_iterations = 0.035, 67    # Optimization hyperparameters
    optimizer = torch.optim.SGD([tensor], lr=learning_rate)  # Stochastic Gradient Descent
    loss_values = []  # List to store loss at each step

    for _ in range(num_iterations):
        optimizer.zero_grad()          # Clear existing gradients
        current_loss = f(tensor)       # Compute loss
        current_loss.backward()        # Backpropagate to compute gradients
        optimizer.step()               # Update tensor with gradients
        loss_values.append(current_loss.item())  # Store current loss
    return tensor

def optimize_torch_fun2(f):
    """
    Minimizes a PyTorch-based function f(x), where x is a tensor of shape (10,).
    Returns the optimized tensor x.
    """
    tensor = torch.rand(10, requires_grad=True)  # Initial vector of shape (10,)
    learning_rate, num_iterations = 0.031, 75
    optimizer = torch.optim.SGD([tensor], lr=learning_rate)
    loss_values = []

    for _ in range(num_iterations):
        optimizer.zero_grad()
        current_loss = f(tensor)
        current_loss.backward()
        optimizer.step()
        loss_values.append(current_loss.item())
    return tensor

def optimize_tf_fun1(f):
    """
    Minimizes a TensorFlow-based function f(x), where x is a tf.Variable of shape (2,).
    Returns the optimized variable x.
    """
    variable = tf.Variable(tf.random.uniform((2,), -1, 1))  # Initial variable between -1 and 1
    learning_rate, num_iterations = 0.032, 70
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_values = []

    for _ in range(num_iterations):
        with tf.GradientTape() as tape:    # Record operations for automatic differentiation
            current_loss = f(variable)
        gradients = tape.gradient(current_loss, variable)  # Compute gradients
        optimizer.apply_gradients([(gradients, variable)]) # Update variable
        loss_values.append(current_loss.numpy())           # Save loss
    return variable

def optimize_tf_fun2(f):
    """
    Minimizes a TensorFlow-based function f(x), where x is a tf.Variable of shape (10,).
    Returns the optimized variable x.
    """
    variable = tf.Variable(tf.random.uniform((10,), -1, 1))
    learning_rate, num_iterations = 0.0302, 97
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_values = []

    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            current_loss = f(variable)
        gradients = tape.gradient(current_loss, [variable])
        optimizer.apply_gradients(zip(gradients, [variable]))
        loss_values.append(current_loss.numpy())
    return variable

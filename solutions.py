import torch
import tensorflow as tf
import numpy as np

def optimize_torch_fun1(f):
    """
    Fast optimization of a PyTorch function with 2D input.
    """
    x = torch.rand(2, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.035)
    for _ in range(67):
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        opt.step()
    return x

def optimize_torch_fun2(f):
    """
    Fast optimization of a PyTorch function with 10D input.
    """
    x = torch.rand(10, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.031)
    for _ in range(75):
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        opt.step()
    return x

def optimize_tf_fun1(f):
    """
    Fast optimization of a TensorFlow function with 2D input.
    """
    x = tf.Variable(tf.random.uniform((2,), -1, 1))
    opt = tf.keras.optimizers.Adam(learning_rate=0.032)
    for _ in range(70):
        with tf.GradientTape() as tape:
            loss = f(x)
        grads = tape.gradient(loss, x)
        opt.apply_gradients([(grads, x)])
    return x

def optimize_tf_fun2(f):
    """
    Fast optimization of a TensorFlow function with 10D input.
    """
    x = tf.Variable(tf.random.uniform((10,), -1, 1))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0302)
    for _ in range(97):
        with tf.GradientTape() as tape:
            loss = f(x)
        grads = tape.gradient(loss, [x])
        opt.apply_gradients(zip(grads, [x]))
    return x


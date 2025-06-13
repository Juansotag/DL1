import torch
import tensorflow as tf
import numpy as np

def optimize_torch_fun1(f):
    """
    Finds arg min f

    Args:
        f: a function with torch operators only that receives a torch tensor of shape (2, ) and will evalue to a float

    Return: torch tensor of shape (2, )
    """
    x = torch.rand(2, requires_grad=True)
    opt = torch.optim.SGD([x], lr=0.035)
    for _ in range(67):
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        opt.step()
    return torch.tensor(x.detach().numpy())  # Returns shape (2,) without gradient

def optimize_torch_fun2(f):
    """
    Finds arg min f

    Args:
        f: a function with torch operators only that receives a torch tensor of shape (10, ) and will evalue to a float
    
    Return: torch tensor of shape (10, )
    """
    x = torch.rand(10, requires_grad=True)
    opt = torch.optim.SGD([x], lr=0.031)
    for _ in range(75):
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        opt.step()
    return torch.tensor(x.detach().numpy())  # Returns shape (10,) without gradient

def optimize_tf_fun1(f):
    """
    Finds arg min f

    Args:
        f: a function with tensorflow operators only that receives a tensorflow Variable of shape (2, ) and will evalue to a float
    
    Return: tensorflow Variable of shape (2, )
    """
    x = tf.Variable(tf.random.uniform((2,), -1, 1))
    opt = tf.keras.optimizers.SGD(learning_rate=0.032)
    for _ in range(70):
        with tf.GradientTape() as tape:
            loss = f(x)
        grads = tape.gradient(loss, x)
        opt.apply_gradients([(grads, x)])
    return tf.Variable(initial_value=x.numpy(), dtype=tf.float32)  # Ensures return is tf.Variable

def optimize_tf_fun2(f):
    """
    Finds arg min f

    Args:
        f: a function with tensorflow operators only that receives a tensorflow Variable of shape (10, ) and will evalue to a float
    
    Return: tensorflow Variable of shape (10, )
    """
    x = tf.Variable(tf.random.uniform((10,), -1, 1))
    opt = tf.keras.optimizers.SGD(learning_rate=0.0302)
    for _ in range(97):
        with tf.GradientTape() as tape:
            loss = f(x)
        grads = tape.gradient(loss, [x])
        opt.apply_gradients(zip(grads, [x]))
    return tf.Variable(initial_value=x.numpy(), dtype=tf.float32)  # Ensures return is tf.Variable

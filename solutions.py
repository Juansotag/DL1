import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

def optimize_torch_fun1(f):

    tensor = torch.rand(2, requires_grad=True) 
    learning_rate, num_iterations = 0.035, 67   
    optimizer = torch.optim.Adam([tensor], lr=learning_rate) 
    loss_values = []  

    for _ in range(num_iterations):
        optimizer.zero_grad()          
        current_loss = f(tensor)       
        current_loss.backward()       
        optimizer.step()               
        loss_values.append(current_loss.item())  
    return tensor

def optimize_torch_fun2(f):

    tensor = torch.rand(10, requires_grad=True)  
    learning_rate, num_iterations = 0.031, 75
    optimizer = torch.optim.Adam([tensor], lr=learning_rate)
    loss_values = []

    for _ in range(num_iterations):
        optimizer.zero_grad()
        current_loss = f(tensor)
        current_loss.backward()
        optimizer.step()
        loss_values.append(current_loss.item())
    return tensor

def optimize_tf_fun1(f):

    variable = tf.Variable(tf.random.uniform((2,), -1, 1))  
    learning_rate, num_iterations = 0.032, 70
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_values = []

    for _ in range(num_iterations):
        with tf.GradientTape() as tape:    
            current_loss = f(variable)
        gradients = tape.gradient(current_loss, variable)  
        optimizer.apply_gradients([(gradients, variable)])
        loss_values.append(current_loss.numpy())           
    return variable

def optimize_tf_fun2(f):

    variable = tf.Variable(tf.random.uniform((10,), -1, 1))
    learning_rate, num_iterations = 0.0302, 97
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_values = []

    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            current_loss = f(variable)
        gradients = tape.gradient(current_loss, [variable])
        optimizer.apply_gradients(zip(gradients, [variable]))
        loss_values.append(current_loss.numpy())
    return variable

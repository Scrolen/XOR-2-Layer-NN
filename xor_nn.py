import pandas as pd
import numpy as np
from activations import relu, relu_derivative_from_activation, sigmoid, sigmoid_derivative_from_activation
from analysis import plot_accuracy, plot_loss, plot_confusion_matrix
np.random.seed(42)


LEARNING_RATE = 1
N_ITERATIONS = 10000

X_rows = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_rows = np.array([[0], [1], [1], [0]])

a0 = X_rows.T  # a0 is our initial input activation layer
y = y_rows.T
m = y.shape[1] # Number of samples

print("Shape of input 'a0':", a0.shape) # Expected: (2, 4)
print("Shape of labels 'y':", y.shape)   # Expected: (1, 4)

# --- Network Parameters ---
n_hidden_neurons = 4
n_input_features = a0.shape[0] # 2
n_output_neurons = y.shape[0]  # 1

# Layer 1
W1 = np.random.randn(n_hidden_neurons, n_input_features) * 0.1 # Shape: (4, 2)
b1 = np.zeros((n_hidden_neurons, 1))                         # Shape: (4, 1)

# Layer 2
W2 = np.random.randn(n_output_neurons, n_hidden_neurons) * 0.1 # Shape: (1, 4)
b2 = np.zeros((n_output_neurons, 1))                         # Shape: (1, 1)

losses = []
accuracies = []

for i in range(1, N_ITERATIONS+1):

    # Make Prediction - We will perform batch proccessing, performing the calculation for all four samples at once. The result Z1 will be a (4, 4) matrix, where each column contains the hidden layer outputs for the corresponding input sample
        # From Input to Hidden Layer  
        Z1 = np.dot(W1, a0) + b1
        A1 = relu(Z1)  # USE RELU HERE
        # From Hidden Layer to Output Layer
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)  # Final Prediction
    # Calculate Loss
        # loss = (1/m) * np.sum((y - A2)**2)
        epsilon = 1e-8
        loss = - (1/m) * np.sum(y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon))#When you calculate the gradients in the next step (backpropagation), this 1/2 will neatly cancel out a 2 that comes from the derivative of the squared term, making the math cleaner
     
        predictions = (A2 > 0.5)
        accuracy = np.mean(predictions == y)

        losses.append(loss)
        accuracies.append(accuracy)

     # --- Calculate Gradients (Backpropagation) ---
        # Layer 2 (Output)
        dZ2 = A2 - y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Layer 1 (Hidden)
        dZ1 = np.dot(W2.T, dZ2) * relu_derivative_from_activation(A1) # USE RELU DERIVATIVE
        dW1 = (1/m) * np.dot(dZ1, a0.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)


    # Gradient Descent
        W1 = W1 - LEARNING_RATE * dW1
        b1 = b1 - LEARNING_RATE * db1

        W2 = W2 - LEARNING_RATE * dW2
        b2 = b2 - LEARNING_RATE * db2

# Plotting Data
plot_loss(losses)
plot_accuracy(accuracies)
# ==================================
# TESTING THE TRAINED MODEL
# ==================================
print("\n------ TRAINING COMPLETE ------")
print(f"Final Loss: {loss:.4f}")

# Perform a final forward pass with the trained weights
Z1 = np.dot(W1, a0) + b1
A1 = relu(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
print("\nFinal Predictions (raw):")
print(A2)

# Convert raw predictions to 0s and 1s
predictions = (A2 > 0.5)

print("\nFinal Predictions (binary):")
print(predictions)

print("\nTrue Values:")
print(y)

plot_confusion_matrix(y, predictions, class_names=['0', '1'])
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_loss(losses):
    """Plots the training loss over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    print("Saved loss curve to 'training_loss_curve.png'")

def plot_accuracy(accuracies):
    """Plots the training accuracy over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, color='green')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1) # Set y-axis from 0 to 110%
    plt.grid(True)
    plt.savefig('training_accuracy_curve.png')
    print("Saved accuracy curve to 'training_accuracy_curve.png'")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Calculates and plots a confusion matrix."""
    # Ensure y_true and y_pred are 1D arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to 'confusion_matrix.png'")


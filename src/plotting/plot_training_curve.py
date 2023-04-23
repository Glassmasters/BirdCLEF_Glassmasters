import matplotlib.pyplot as plt


def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the training and validation losses and accuracies over the epochs.

    This function creates two side-by-side plots. The left plot shows the training
    and validation losses over the epochs, and the right plot shows the training
    and validation accuracies over the epochs. Both plots use the same x-axis,
    representing the number of epochs.

    Args:
    train_losses (list): A list of training loss values for each epoch.
    val_losses (list): A list of validation loss values for each epoch.
    train_accuracies (list): A list of training accuracy values for each epoch.
    val_accuracies (list): A list of validation accuracy values for each epoch.

    Returns:
    None: Displays the generated plots using matplotlib.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

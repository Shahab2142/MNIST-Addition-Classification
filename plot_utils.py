from common_imports import plt, np, confusion_matrix, ConfusionMatrixDisplay

def plot_training_history(history: dict) -> None:
    """
    Plot the training and validation loss and accuracy curves over epochs.

    Args:
        history (dict): Dictionary containing training and validation loss and accuracy.
                        Keys:
                            - "train_loss": List of training loss values over epochs.
                            - "val_loss": List of validation loss values over epochs.
                            - "train_acc": List of training accuracy percentages over epochs.
                            - "val_acc": List of validation accuracy percentages over epochs.

    Saves:
        - "train_val_loss.pdf": PDF plot of training and validation loss over epochs.
        - "train_val_accur.pdf": PDF plot of training and validation accuracy over epochs.
    """

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("train_val_loss.pdf", format="pdf")
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("train_val_accur.pdf", format="pdf")
    plt.show()

def plot_confusion_matrix(predictions, ground_truths, class_names):
    """
    Plot the confusion matrix comparing predictions and ground truths with enhanced readability for LaTeX integration.

    Args:
        predictions (list or np.ndarray): Predicted class labels.
        ground_truths (list or np.ndarray): True class labels.
        class_names (list): List of class names corresponding to the labels.

    Saves:
        - "confusion.pdf": PDF plot of the confusion matrix with class labels and dynamically adjusted font sizes.

    Notes:
        - Font sizes for annotations are dynamically scaled based on the number of classes for better readability.
        - Zero entries in the confusion matrix are hidden to improve clarity.
    """

    # Compute confusion matrix
    cm = confusion_matrix(ground_truths, predictions)
    num_classes = len(class_names)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size for readability
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.0f')  # Format as integers

    # Dynamically adjust font size for annotations
    max_font_size = 20  # Set a maximum font size
    min_font_size = 8   # Set a minimum font size
    optimal_font_size = max(min_font_size, min(max_font_size, 200 / num_classes))  # Scale font size based on number of classes

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_obj = ax.texts[i * cm.shape[1] + j]
            text_obj.set_fontsize(optimal_font_size)  # Adjust font size dynamically
            if cm[i, j] == 0:
                text_obj.set_visible(False)  # Hides the zero values

    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)
    plt.xticks(fontsize=13, rotation=45)
    plt.yticks(fontsize=13)
    plt.tight_layout()  # Avoid clipping of labels
    plt.savefig("confusion.pdf", format="pdf", bbox_inches="tight")
    plt.show()




def compute_and_plot_per_class_accuracy(predictions, ground_truths, num_classes):
    """
    Compute and plot the accuracy for each class with enhanced readability for LaTeX integration.

    Args:
        predictions (list or np.ndarray): Predicted class labels.
        ground_truths (list or np.ndarray): True class labels.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: Array containing per-class accuracy percentages.

    Saves:
        - "per_class.pdf": PDF bar chart of accuracy percentages for each class.

    Notes:
        - Accuracy for each class is calculated as:
            (number of correct predictions for the class) / (total instances of the class) * 100.
        - Classes are displayed on the x-axis, and accuracy percentages are displayed on the y-axis.
    """

    # Ensure the correct number of class names
    class_names = [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("Number of class names must match num_classes.")

    # Initialize arrays to count correct and total instances for each class
    correct_per_class = np.zeros(num_classes, dtype=int)
    total_per_class = np.zeros(num_classes, dtype=int)

    # Compute counts for each class
    for pred, true in zip(predictions, ground_truths):
        total_per_class[true] += 1
        if pred == true:
            correct_per_class[true] += 1

    # Compute accuracy as a percentage
    accuracy_per_class = 100 * correct_per_class / total_per_class

    # Plot per-class accuracy
    plt.figure(figsize=(6, 6))
    plt.bar(class_names, accuracy_per_class, color='skyblue')
    plt.xlabel('Classes', fontsize=16)
    plt.ylabel('Accuracy (\%)', fontsize=16)
    plt.title('Accuracy Per Class', fontsize=20)
    plt.xticks(fontsize=17, rotation=45, ha='center')
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig("per_class.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return accuracy_per_class

def visualize_original_and_augmented_data(original_images: np.ndarray, augmented_images: np.ndarray, n: int = 5, image_shape: tuple = (56, 28)) -> None:
    """
    Visualize original and augmented images side by side horizontally.

    Args:
        original_images (np.ndarray): Array of original images. Each image should be 2D or 1D that can be reshaped to `image_shape`.
        augmented_images (np.ndarray): Array of augmented images with the same format as `original_images`.
        n (int): Number of image pairs to visualize. Default is 5.
        image_shape (tuple): Shape of each image as (height, width). Default is (56, 28).

    Saves:
        - "augmented.pdf": PDF visualization of original and augmented images side by side.

    Notes:
        - Each pair consists of one original image and its corresponding augmented version.
        - Images are displayed in grayscale with the titles "Original" and "Augmented" for clarity.
    """

    # Plot original and augmented images horizontally
    fig, axes = plt.subplots(1, 2 * n, figsize=(2 * n * 2, 4))
    for i in range(n):
        # Original image
        axes[2 * i].imshow(original_images[i].reshape(image_shape), cmap="gray")
        axes[2 * i].set_title(f"Original {i + 1}")
        axes[2 * i].axis("off")

        # Augmented image
        axes[2 * i + 1].imshow(augmented_images[i].reshape(image_shape), cmap="gray")
        axes[2 * i + 1].set_title(f"Augmented {i + 1}")
        axes[2 * i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("augmented.pdf", format="pdf")

    plt.show()


def plot_tsne(data_2d, labels, title):
    """
    Plot a 2D scatter plot of t-SNE embeddings.

    Args:
        data_2d (np.ndarray): 2D array of t-SNE embeddings with shape (num_samples, 2).
        labels (np.ndarray or list): Class labels corresponding to each data point.
        title (str): Title of the plot.

    Saves:
        - "tsne.pdf": PDF plot of the t-SNE embeddings with class-based coloring.

    Notes:
        - Points are colored based on their class labels using the `tab10` colormap.
        - Useful for visualizing high-dimensional data in two dimensions to understand class separability.
    """

    # Scatter plot with class labels as colors
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, label="Class")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(alpha=0.3)
    plt.savefig("tsne.pdf", format="pdf")
    plt.show()

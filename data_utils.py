from common_imports import np, os, random, torch, TensorDataset, DataLoader, MNIST, ToTensor, Compose, RandomRotation, ColorJitter, RandomErasing, TSNE, silhouette_score, davies_bouldin_score

def set_reproducibility(seed: int = 42) -> None:
    """
    Set seeds for reproducibility across random number generators in various libraries.

    Args:
        seed (int): Seed value for random number generators. Default is 42.

    Effects:
        - Ensures deterministic behavior in NumPy, Python `random`, and PyTorch.
        - Enables deterministic algorithms in PyTorch, ensuring reproducible results.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_mnist_data() -> tuple:
    """
    Load the MNIST dataset and return normalized training and test data.

    Returns:
        tuple:
            - (x_train, y_train): Training images and labels as NumPy arrays, normalized to [0, 1].
            - (x_test, y_test): Test images and labels as NumPy arrays, normalized to [0, 1].

    Notes:
        - MNIST images are loaded as 28x28 grayscale images.
        - Labels are integers representing class indices (0-9).
    """

    mnist_train = MNIST(root="./data", train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(root="./data", train=False, download=True, transform=ToTensor())

    x_train, y_train = mnist_train.data.numpy(), mnist_train.targets.numpy()
    x_test, y_test = mnist_test.data.numpy(), mnist_test.targets.numpy()

    return (x_train / 255.0, y_train), (x_test / 255.0, y_test)


def create_pairs(images: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Create paired images by vertically stacking and flipping them for training.

    Args:
        images (np.ndarray): Array of input images with shape (num_samples, height, width).
        labels (np.ndarray): Array of corresponding labels for the images.

    Returns:
        tuple:
            - paired_images (np.ndarray): Array of paired images created by stacking and flipping.
            - paired_labels (np.ndarray): Array of summed labels for the paired images.

    Notes:
        - Pairing is done randomly by shuffling indices.
        - Each pair is vertically stacked twice: original order and flipped order.
    """

    num_pairs = len(images) // 2
    paired_images, paired_labels = [], []
    indices = np.random.permutation(len(images))
    for i in range(num_pairs):
        img1, img2 = images[indices[2 * i]], images[indices[2 * i + 1]]
        label1, label2 = labels[indices[2 * i]], labels[indices[2 * i + 1]]
        paired_images.extend([np.vstack((img1, img2)), np.vstack((img2, img1))])
        paired_labels.extend([label1 + label2, label1 + label2])
    return np.array(paired_images), np.array(paired_labels)


def add_gaussian_noise(image: torch.Tensor) -> torch.Tensor:
    """
    Add Gaussian noise to an image tensor.

    Args:
        image (torch.Tensor): Input image tensor with values in [0, 1].

    Returns:
        torch.Tensor: Image tensor with added Gaussian noise, clamped to [0, 1].

    Notes:
        - Noise is sampled from a normal distribution with a standard deviation of 0.05.
        - The output image is clamped to ensure values stay within the valid range.
    """

    noise = torch.randn(image.size()) * 0.05
    return torch.clamp(image + noise, 0.0, 1.0)


def add_salt_and_pepper(image: torch.Tensor) -> torch.Tensor:
    """
    Add salt-and-pepper noise to an image tensor.

    Args:
        image (torch.Tensor): Input image tensor with values in [0, 1].

    Returns:
        torch.Tensor: Image tensor with added salt-and-pepper noise.

    Notes:
        - Pixels are set to 0 (pepper) or 1 (salt) with probabilities 0.01 each.
        - This simulates image degradation often seen in noisy environments.
    """

    mask = torch.rand_like(image)
    image[mask < 0.01] = 0.0
    image[mask > 0.99] = 1.0
    return image


def augment_data(images: np.ndarray, labels: np.ndarray, add_noise: bool = True) -> tuple:
    """
    Apply data augmentation and optionally add noise to input images.

    Args:
        images (np.ndarray): Array of input images to augment.
        labels (np.ndarray): Array of corresponding labels for the images.
        add_noise (bool): Whether to add Gaussian and salt-and-pepper noise to the images. Default is True.

    Returns:
        tuple:
            - augmented_images (np.ndarray): Array of augmented images.
            - labels (np.ndarray): Array of corresponding labels (unchanged).

    Notes:
        - Augmentations include random rotations, color jitter, and random erasing.
        - Gaussian and salt-and-pepper noise are applied stochastically if `add_noise` is True.
    """

    augmented_transform = Compose([
        RandomRotation(degrees=30),
        ColorJitter(brightness=0.3, contrast=0.3),
        RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    augmented_images = []
    for img in images:
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        augmented_img = augmented_transform(img_tensor).squeeze(0)
        if add_noise:
            if random.random() < 0.5:
                augmented_img = add_gaussian_noise(augmented_img)
            if random.random() < 0.5:
                augmented_img = add_salt_and_pepper(augmented_img)
        augmented_images.append(augmented_img.numpy())

    return np.array(augmented_images), labels


def prepare_datasets(
    mode: str = "56x28",
    data_percentage: float = 1.0,
    validation_size: int = 8000,
    test_size: int = 2000,
) -> tuple:
    """
    Prepare training, validation, and testing datasets for training a neural network.

    Args:
        mode (str): Dataset structure mode, either "28x28" or "56x28". Default is "56x28".
        data_percentage (float): Fraction of training data to retain (0 < data_percentage <= 1.0). Default is 1.0.
        validation_size (int): Number of samples for validation. Default is 8000.
        test_size (int): Number of samples for testing. Default is 2000.

    Returns:
        tuple:
            - train_dataset (TensorDataset): Training dataset as a PyTorch TensorDataset.
            - val_dataset (TensorDataset): Validation dataset as a PyTorch TensorDataset.
            - test_dataset (TensorDataset): Testing dataset as a PyTorch TensorDataset.

    Raises:
        ValueError: If `mode` is not "28x28" or "56x28".

    Notes:
        - Augmentation is applied to the training data.
        - In "56x28" mode, paired images are created by vertically stacking two 28x28 images.
    """

    # Load MNIST data
    (original_train_images, original_train_labels), (original_test_images, original_test_labels) = load_mnist_data()

    # Reduce training data based on the data_percentage
    num_train_samples = int(len(original_train_images) * data_percentage)
    filtered_train_images, filtered_train_labels = original_train_images[:num_train_samples], original_train_labels[:num_train_samples]

    if mode == "28x28":
        # Augment and prepare for 28x28
        augmented_train_images, augmented_train_labels = augment_data(filtered_train_images, filtered_train_labels)
        final_train_images = np.vstack((filtered_train_images, augmented_train_images))
        final_train_labels = np.concatenate((filtered_train_labels, augmented_train_labels))
        validation_images = original_test_images[:validation_size]
        validation_labels = original_test_labels[:validation_size]
        test_images = original_test_images[validation_size:validation_size + test_size]
        test_labels = original_test_labels[validation_size:validation_size + test_size]
        flattened_train_images = final_train_images.reshape(len(final_train_images), -1)
        flattened_validation_images = validation_images.reshape(len(validation_images), -1)
        flattened_test_images = test_images.reshape(len(test_images), -1)
    elif mode == "56x28":
        # Augment and prepare for 56x28 paired data
        paired_train_images, paired_train_labels = create_pairs(filtered_train_images, filtered_train_labels)
        augmented_paired_images, augmented_paired_labels = augment_data(paired_train_images, paired_train_labels)
        final_train_images = np.vstack((paired_train_images, augmented_paired_images))
        final_train_labels = np.concatenate((paired_train_labels, augmented_paired_labels))
        paired_test_images, paired_test_labels = create_pairs(original_test_images, original_test_labels)
        validation_images, validation_labels = paired_test_images[:validation_size], paired_test_labels[:validation_size]
        test_images, test_labels = paired_test_images[validation_size:validation_size + test_size], paired_test_labels[validation_size:validation_size + test_size]
        flattened_train_images = final_train_images.reshape(len(final_train_images), -1)
        flattened_validation_images = validation_images.reshape(len(validation_images), -1)
        flattened_test_images = test_images.reshape(len(test_images), -1)
    else:
        raise ValueError("Invalid mode! Choose '28x28' or '56x28'.")

    # Convert to TensorDataset
    train_dataset = TensorDataset(torch.tensor(flattened_train_images, dtype=torch.float32), torch.tensor(final_train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(flattened_validation_images, dtype=torch.float32), torch.tensor(validation_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(flattened_test_images, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size: int):
    """
    Create DataLoader objects for training, validation, and testing datasets.

    Args:
        train_dataset (TensorDataset): Dataset for training.
        val_dataset (TensorDataset): Dataset for validation.
        test_dataset (TensorDataset): Dataset for testing.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for training.
            - val_loader (DataLoader): DataLoader for validation.
            - test_loader (DataLoader): DataLoader for testing.

    Notes:
        - The number of workers for data loading is set to `os.cpu_count() - 1` for parallel processing.
        - The random seed ensures reproducibility when shuffling the data.
    """

    generator = torch.Generator()
    generator.manual_seed(42)  # Ensure reproducibility for shuffling

    num_workers = os.cpu_count() - 1  # Use one less than the total number of CPU cores
    print(f"Using {num_workers} workers for DataLoader.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader



def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings (outputs of the embedding layer) from a trained model for a given dataset.

    Args:
        model (torch.nn.Module): Trained neural network model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to use for computation (e.g., CPU or GPU).

    Returns:
        tuple:
            - embeddings (np.ndarray): 2D array of embeddings for the dataset.
            - labels (np.ndarray): Array of corresponding labels for the embeddings.

    Notes:
        - The model is set to evaluation mode during this process.
        - Embeddings are extracted from the last hidden layer before the output layer.
    """

    model.eval()
    embeddings = []
    labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            embedding_output = model.embedding_layer_output(x_batch)
        embeddings.append(embedding_output.cpu().numpy())
        labels.append(y_batch.numpy())
    return np.vstack(embeddings), np.concatenate(labels)


def compute_tsne_metrics(data_2d, labels):
    """
    Compute clustering metrics for 2D t-SNE embeddings.

    Args:
        data_2d (np.ndarray): 2D t-SNE embeddings.
        labels (np.ndarray): Ground truth labels for the data.

    Returns:
        tuple:
            - silhouette_score (float): Metric indicating how similar samples in a cluster are (higher is better).
            - davies_bouldin_index (float): Metric indicating the ratio of intra-cluster to inter-cluster distances (lower is better).

    Notes:
        - Silhouette score ranges from -1 (poor) to 1 (ideal).
        - Davies-Bouldin index is lower for better clustering performance.
    """

    silhouette = silhouette_score(data_2d, labels)
    davies_bouldin = davies_bouldin_score(data_2d, labels)
    return silhouette, davies_bouldin
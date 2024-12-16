from utils.common_imports import torch, nn, ReduceLROnPlateau, optuna

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class FullyConnectedNN(nn.Module):
    """
    Fully Connected Neural Network with customizable layers and activations.

    Args:
        input_size (int): Size of the input features.
        hidden_sizes (list): List containing the number of neurons for each hidden layer.
        output_size (int): Number of output classes or dimensions.
        dropout_rate (float): Dropout probability for regularization.
        activation_function (str): Activation function for the hidden layers (e.g., "ReLU", "Tanh").

    Attributes:
        hidden_layers (nn.ModuleList): List of sequential hidden layers with linear transformations,
            activation functions, batch normalization, and dropout.
        output_layer (nn.Linear): Final linear transformation layer mapping to the output size.
    """

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout_rate: float, activation_function: str = "ReLU"):
        super(FullyConnectedNN, self).__init__()
        activation_functions = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
            "LeakyReLU": nn.LeakyReLU
        }
        if activation_function not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_function}")

        # Initialize hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_size, hidden_size),
                    activation_functions[activation_function](),
                    nn.BatchNorm1d(hidden_size),  # Batch normalization for stability
                    nn.Dropout(dropout_rate)  # Dropout for regularization
                )
            )
            in_size = hidden_size  # Update input size for the next layer

        # Initialize output layer
        self.output_layer = nn.Linear(in_size, output_size)

        # Apply weight initialization
        self.apply(self._initialize_weights)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)  # Final output layer
        return x

    def embedding_layer_output(self, x):
        """
        Extract the output from the embedding layer (the last hidden layer).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the embedding layer.
        """
        for layer in self.hidden_layers:
            x = layer(x)
        return x  # Output before the final layer

    @staticmethod
    def _initialize_weights(module):
        """
        Initialize weights for linear layers using Xavier uniform initialization.

        Args:
            module (nn.Module): A module to initialize. If it is a linear layer (nn.Linear),
                its weights are initialized using Xavier uniform distribution, and its bias is set to zero.
        """

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs: int, patience: int = 3, trial=None):
    """
    Train a neural network with early stopping, learning rate scheduling, and optional Optuna pruning.

    Args:
        model (FullyConnectedNN): Neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function to optimize during training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to adjust the learning rate based on validation loss.
        epochs (int): Maximum number of training epochs.
        patience (int, optional): Number of epochs to wait for validation loss improvement before stopping early (default: 3).
        trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization (default: None).

    Returns:
        dict: Training history containing:
            - "train_loss": List of training loss values for each epoch.
            - "val_loss": List of validation loss values for each epoch.
            - "train_acc": List of training accuracy percentages for each epoch.
            - "val_acc": List of validation accuracy percentages for each epoch.

    Raises:
        optuna.TrialPruned: If the trial does not meet the pruning criteria.
    """

    model = model.to(device)  # Move model to the correct device

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss, correct, total = 0, 0, 0

        # Training loop
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to device
            optimizer.zero_grad()  # Reset gradients
            outputs = model(x_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        # Compute training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Compute validation metrics
        val_metrics = compute_metrics(model, val_loader, criterion)
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]

        # Report to Optuna
        if trial is not None:
            trial.report(val_acc, epoch)

            # Prune the trial if validation accuracy is less than 50% after the first epoch
            if epoch == 0 and val_acc < 50.0:
                print(f"Pruning trial {trial.number}: Validation accuracy {val_acc:.2f}% is below threshold after first epoch.")
                raise optuna.TrialPruned()

            if trial.should_prune():
                print(f"Pruning trial {trial.number}: Validation performance insufficient.")
                raise optuna.TrialPruned()

        # Update scheduler
        scheduler.step(val_loss)

        # Record metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return history

def compute_metrics(model, data_loader, criterion):
    """
    Compute loss, accuracy, and predictions for a given dataset.

    Args:
        model (FullyConnectedNN): Trained neural network model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        criterion (nn.Module): Loss function to compute the loss on the dataset.

    Returns:
        dict: Dictionary containing:
            - "loss": Average loss over the dataset.
            - "accuracy": Accuracy percentage over the dataset.
            - "predictions": List of predicted class indices for all samples in the dataset.
            - "ground_truths": List of true class indices for all samples in the dataset.
    """ 

    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to device
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(predicted.cpu().numpy())  # Move back to CPU for storage
            all_targets.extend(y_batch.cpu().numpy())

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": 100 * correct / total,
        "predictions": all_preds,
        "ground_truths": all_targets,
    }

def initialize_optimizer(optimizer_name: str, model: FullyConnectedNN, learning_rate: float):
    """
    Initialize the optimizer for training.

    Args:
        optimizer_name (str): Name of the optimizer to use ("Adam", "SGD", "RMSprop").
        model (FullyConnectedNN): Neural network model whose parameters will be optimized.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: Initialized optimizer for training.

    Raises:
        ValueError: If the specified optimizer_name is not supported.
    """

    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
        "RMSprop": torch.optim.RMSprop
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizers[optimizer_name](model.parameters(), lr=learning_rate)

def initialize_nn_and_training_components(config, input_size, output_size):
    """
    Initialize the neural network and its associated training components, including the criterion, optimizer, and scheduler.

    Args:
        config (dict): Configuration for the neural network and training, including:
            - "hidden_sizes" (list): List of neurons in each hidden layer.
            - "dropout_rate" (float): Dropout probability for regularization.
            - "activation_function" (str): Activation function for the hidden layers.
            - "optimizer" (str): Name of the optimizer ("Adam", "SGD", "RMSprop").
            - "lr" (float): Learning rate for the optimizer.
        input_size (int): Number of input features.
        output_size (int): Number of output classes or dimensions.

    Returns:
        tuple: (model, criterion, optimizer, scheduler), where:
            - model (FullyConnectedNN): Initialized neural network.
            - criterion (nn.Module): Cross-entropy loss function.
            - optimizer (torch.optim.Optimizer): Optimizer for training.
            - scheduler (torch.optim.lr_scheduler): ReduceLROnPlateau scheduler for adjusting the learning rate.
    """

    # Initialize the model
    model = FullyConnectedNN(
        input_size=input_size,
        hidden_sizes=config["hidden_sizes"],
        output_size=output_size,
        dropout_rate=config["dropout_rate"],
        activation_function=config["activation_function"]
    ).to(device)  # Move model to device
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = initialize_optimizer(config["optimizer"], model, config["lr"])
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2)
    
    return model, criterion, optimizer, scheduler

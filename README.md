# **Neural Network Performance Benchmarking and Optimizations**

## **Project Overview**
This project systematically investigates the design and training of neural network models for an addition classification tasks using the MNIST dataset. The primary goals include analyzing the impact of architecture, optimizers, and hyperparameters on model performance and convergence, as well as implementing classical machine learning models (Random Forests and SVMs) for comparison.

---

## **Project Structure**

The project is organized into the following core components:

### 1. **`main_neural_net.ipynb`**
   - Builds and trains **fully connected neural networks (FCNN)** on paired and augmented MNIST datasets.
   - Implements **Optuna** for hyperparameter optimization, focusing on:
     - Dynamic architecture search (number of layers, neurons per layer, dropout rates).
     - Optimizer choice (`Adam`, `SGD`, `RMSprop`).
   - Generates key metrics such as training/validation accuracy and loss curves.
   - Extracts embeddings and evaluates class separability using **t-SNE** and silhouette scores.
   - **Key Features**:
     - Custom data augmentation (Gaussian noise, salt-and-pepper noise, and transformations).
     - Visualization of:
       - Training/Validation Loss.
       - Confusion Matrices.
       - Per-Class Accuracy.
       - t-SNE embeddings.

---

### 2. **`svm_rf.ipynb`**
   - Implements **Random Forest** and **Support Vector Machines (SVM)** for multi-class classification.
   - Utilizes **PCA** for dimensionality reduction, retaining 95% variance.
   - Performs **hyperparameter tuning** for Random Forest using Optuna.
   - **Key Features**:
     - Random Forest Hyperparameter Search:
       - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
     - SVM:
       - Uses a linear kernel for high-dimensional MNIST data.

---

### 3. **`weak_linear.ipynb`**
   - Trains and evaluates **weak linear classifiers** on paired MNIST datasets (`56x28` and `28x28`).
   - Models are trained for different dataset sizes to observe underfitting/overfitting trends.
   - **Key Features**:
     - Evaluation of training and validation loss over different data sizes.
     - Sequential predictions for `28x28` models on `56x28` test data.
     - **Timing metrics** to track training duration for each dataset size.

---

### 4. **`experiments.ipynb`**
   - Systematically explores neural network architectures and hyperparameters to identify optimal configurations.
   - **Experiments Conducted**:
     1. **Effect of Layers and Neurons**:
        - Analysis using 3D visualizations.
     2. **Depth-to-Width Trade-off**:
        - Validation loss comparison for decreasing layer sizes.
     3. **Dropout Rate and Activation Function**:
        - Impact of dropout rates with `ReLU` and `Tanh` activations.
     4. **Optimizer Convergence**:
        - Comparison of `Adam`, `SGD`, and `RMSprop` optimizers on convergence speed and loss reduction.
     5. **Batch Size vs. Learning Rate**:
        - Heatmaps and visualizations for optimal batch size and learning rate combinations.

---

### 5. **Utility Files**

- **`data_utils.py`**:
   - Functions for data preparation, augmentation, and reproducibility.
   - Includes methods for pairing and augmenting MNIST images.
   - Implements custom DataLoader utilities.

- **`model_utils.py`**:
   - Core functions for initializing neural networks, optimizers, and schedulers.
   - Implements training loops with early stopping, learning rate scheduling, and Optuna pruning.
   - Provides evaluation utilities for accuracy and loss.

- **`plot_utils.py`**:
   - Visualization functions for:
     - Training/Validation Loss and Accuracy.
     - Confusion Matrices.
     - Per-Class Accuracy.
     - t-SNE Embeddings.

- **`common_imports.py`**:
   - Centralized imports for libraries like PyTorch, Matplotlib, Optuna, and Scikit-learn.


This project is licensed under the MIT License. See LICENSE for more details.


Declaration of Autogenerative Tools

In this project, I used GitHub Copilot as a coding assistant. I mainly used it to generate the plots in Figures, from 1 to 10, alongside the generation of the 6 tables. I also used it within my code to create comments, docstring my functions, debug and format markdown cells explaining the different sections in the notebook.

In regards to the report, I used the Overleaf AI which uses ChatGPT to help write the mathematics in latex format, and also used it to correctly format various sections of the report as I am quite new to LaTeX.



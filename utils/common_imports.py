# Numerical computation
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn  # Neural network utilities

# Optimization and learning rate scheduling
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Curve fitting and optimization
from scipy.optimize import curve_fit

# Dataset and transformations
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    Compose,
    RandomRotation,
    ColorJitter,
    RandomErasing,
)

# Metrics and evaluation
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score,
    davies_bouldin_score,
)

# Dimensionality reduction
from sklearn.manifold import TSNE

# File I/O and utility
import os
from itertools import combinations

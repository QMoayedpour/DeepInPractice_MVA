# import numpy as np

import torch

from torchvision import datasets, transforms, models
from torch.functional import F
import torch.nn as nn
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics import Metric

def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is 
    the rest of the model, it is not necessary to recompute $g(x)$ at 
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and 
    create a new dataset 
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation
    
    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    raise NotImplementedError("Precompute the features of the dataset using the model")


class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        # <YOUR CODE>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <YOUR CODE>
        raise NotImplementedError("Implement the forward pass of the LastLayer module")


class FinalModel(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        # <YOUR CODE>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <YOUR CODE>
        raise NotImplementedError("Implement the forward pass of the LastLayer module")

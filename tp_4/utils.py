# import numpy as np

import torch

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
from torch.functional import F
import torch.nn as nn
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics import Metric


def precompute_features(
    model: models.ResNet, 
    dataset: torch.utils.data.Dataset, 
    device: torch.device
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
    model = model.to(device)
    model.eval()  # to go faster (dont compute the grads) and to set dropout as 0

    get_feats = torch.nn.Sequential(*list(model.children())[:-1]) # we remove the last layer (the one we wanna train)

    get_feats.to(device)

    feats = []
    labs = []

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feat = get_feats(images)
            feat = feat.view(feat.size(0), -1)
            feats.append(feat.cpu())
            labs.append(labels.cpu())

    features = torch.cat(feats)
    labels = torch.cat(labs)

    return TensorDataset(features, labels)


class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.fc(x)


class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

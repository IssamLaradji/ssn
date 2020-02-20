import torch

from torch import nn
from torch.nn import functional as F
import math
import torchvision.models as models


def get_model(model_name, train_set=None):
    if model_name == 'linear':
        batch = train_set[0]
        model = LinearRegression(input_dim=batch['images'].shape[0], output_dim=1)

    return model

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

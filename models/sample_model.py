import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleModel(nn.Module):
    """
    Explanation
    """
    def __init__(self, h):
        super(SampleModel, self).__init__()

        # Set parameters

        # Check configuration (h)
        self._check_hyperparameters(h)

        # Construct modules
        self.layer = nn.Linear(10, 20)

    def _check_hyperparameters(self, h):
        pass

    def forward(self):
        pass


class SampleDiscriminator(nn.Module):
    """
    Explanation
    """
    def __init__(self, h):
        super(SampleDiscriminator, self).__init__()

        # Set parameters

        # Check configuration (h)
        self._check_hyperparameters(h)

        # Construct modules
        self.layer = nn.Linear(10, 20)

    def _check_hyperparameters(self, h):
        pass

    def forward(self):
        pass


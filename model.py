import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F

class Model(nn.Module):
    """
    Class for defining the model
    """

    def __init__(self):
        super().__init__()
        # Must construct parent model class

        self.hidden = nn.Linear(4, 3)
        # TODO: 3 might not be the correct number of hidden nodes
        self.output = nn.Linear(3, 1)

    def feed_forward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        x = self.output(x)


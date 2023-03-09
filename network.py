import torch.nn as nn
import torch.nn.functional as func

class Network(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(Network, self).__init__()
        # convolution layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layer
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        # pooling layer
        self.pool1 = nn.MaxPool2d(2)


    def forward(self, x):
        # conv1->act->pool->conv2->act->pool->fc1->act->fc2->act->fc3
        x = self.pool1(func.relu(self.conv1(x)))
        x = self.pool1(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
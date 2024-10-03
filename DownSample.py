import torch.nn as NN
from BlueArrow import BlueArrow

class DownSample(NN.Module):
    # Building block of the contractive path of the network.
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convolution = BlueArrow(inChannels, outChannels)
        self.redArrow = NN.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        down = self.convolution(x)
        pooling = self.redArrow(down)
        return down, pooling
import torch.nn as NN
from torch import cat
from BlueArrow import BlueArrow
from AttentionGates import AttentionGates

class UpSample(NN.Module):
    # Building block of the expansive path of the network.
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.greenArrow = NN.ConvTranspose2d(inChannels, inChannels // 2, kernel_size = 2, stride = 2)
        # Implementation of the attention mechanism. 
        # Intermediate channels are chosen for computational efficiency and dimensionality reduction.
        self.attentionGate = AttentionGates(decoderInput = inChannels // 2, encoderInput = inChannels // 2, 
                                            intermediateChannels = inChannels // 4)
        self.convolution = BlueArrow(inChannels, outChannels)
       
    def forward(self, x1, x2):
        x1 = self.greenArrow(x1)
        x2 = self.attentionGate(x1, x2)
        x = cat([x1, x2], dim = 1)
        return self.convolution(x)
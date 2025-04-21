from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU
from ChannelAttention import ChannelAttention

class BlueArrow(Module):
    # Fundamental component of the original U-Net.
    # Refer to [https://arxiv.org/pdf/1505.04597].
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convolutionOperation = Sequential(Conv2d(inChannels, outChannels, kernel_size = 3, padding = 1),
                                               # Batch normalization is not included in the original paper.
                                               # Authors included it in their subsequent work [https://arxiv.org/pdf/1606.06650].
                                               # Here it has thus been implemented.
                                               BatchNorm2d(outChannels),
                                               ReLU(inplace = True),
                                               Conv2d(outChannels, outChannels, kernel_size = 3, padding = 1),
                                               BatchNorm2d(outChannels),
                                               ReLU(inplace = True),
                                               ChannelAttention(outChannels))

    def forward(self, x):
        return self.convolutionOperation(x)
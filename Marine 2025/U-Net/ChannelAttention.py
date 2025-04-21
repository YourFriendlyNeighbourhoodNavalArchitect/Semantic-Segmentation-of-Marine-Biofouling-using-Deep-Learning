from torch.nn import Module, AdaptiveAvgPool2d, Sequential, Conv2d, ReLU, Sigmoid

class ChannelAttention(Module):
    # Channel attention mechanism, that adaptively recalibrates channel-wise feature responses.
    # Refer to [https://arxiv.org/abs/1709.01507v4].
    def __init__(self, channel):
        super().__init__()
        self.squeeze = AdaptiveAvgPool2d(1)
        self.excitation = Sequential(Conv2d(channel, channel // 16, kernel_size = 1, bias = False),
                                     ReLU(inplace = True),
                                     Conv2d(channel // 16, channel, kernel_size = 1, bias = False),
                                     Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)

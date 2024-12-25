import torch.nn as NN
from torch.nn.init import kaiming_normal_, constant_

def initializeWeights(module):
    # Weight initialization in neural networks improves performance and stability.
    # Refer to [https://arxiv.org/pdf/2406.00348v1].
    if isinstance(module, NN.Conv2d) or isinstance(module, NN.ConvTranspose2d):
        # Weights initialized as normally distributed, scaled by the number of output features.
        kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
        if module.bias is not None:
            constant_(module.bias, 0)

    elif isinstance(module, NN.BatchNorm2d):
        # Identity mapping (no effect).
        constant_(module.weight, 1)
        constant_(module.bias, 0)
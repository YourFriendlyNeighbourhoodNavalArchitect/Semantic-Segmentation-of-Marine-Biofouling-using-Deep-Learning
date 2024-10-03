import torch.nn as NN
import torch.nn.init as init

def initializeWeights(module):
    # Weight initialization in neural networks improves performance and stability.
    # Refer to [https://arxiv.org/pdf/2406.00348v1].
    if isinstance(module, NN.Conv2d) or isinstance(module, NN.ConvTranspose2d):
        # Weights initialized as normally distributed, scaled by the number of output features.
        init.kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, NN.BatchNorm2d):
        # Identity mapping (no effect).
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
from torch.nn import BatchNorm2d, Conv2d
from torch.nn.init import constant_, kaiming_normal_

from u_net_models.UNet import UNet


def initializeWeights(module: UNet) -> None:
    # Weight initialization in neural networks improves performance and stability.
    # Refer to [https://arxiv.org/pdf/2406.00348v1].
    if isinstance(module, Conv2d):
        # Weights initialized as normally distributed, scaled by the number of output features.
        kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            constant_(module.bias, 0)

    elif isinstance(module, BatchNorm2d):
        # Identity mapping (no effect).
        constant_(module.weight, 1)
        constant_(module.bias, 0)

import torch
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU, Sequential, Upsample

from u_net_models.BlueArrow import BlueArrow
from u_net_models.SpatialAttention import SpatialAttention


class UpSample(Module):
    # Building block of the expansive path of the network.
    def __init__(self, inChannels: int, outChannels: int) -> None:
        super().__init__()
        # Modern decoder architectures use upsampling blocks, instead of transpose convolutions.
        self.greenArrow = Sequential(
            Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv2d(inChannels, inChannels // 2, kernel_size=3, padding=1),
            BatchNorm2d(inChannels // 2),
            ReLU(inplace=True),
        )
        # Implementation of the spatial attention mechanism.
        # Intermediate channels are chosen for computational efficiency and dimensionality reduction.
        self.attentionGate = SpatialAttention(
            decoderInput=inChannels // 2,
            encoderInput=inChannels // 2,
            intermediateChannels=inChannels // 4,
        )
        self.convolution = BlueArrow(inChannels, outChannels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.greenArrow(x1)
        x2 = self.attentionGate(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        return self.convolution(x)

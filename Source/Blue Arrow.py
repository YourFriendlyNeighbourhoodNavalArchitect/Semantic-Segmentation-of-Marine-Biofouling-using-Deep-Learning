import torch.nn as NN

class BlueArrow(NN.Module):
    # Fundamental component of the original U-Net.
    # Refer to [https://arxiv.org/pdf/1505.04597].
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convolutionOperation = NN.Sequential(
            NN.Conv2d(inChannels, outChannels, kernel_size = 3, padding = 1),
            # Batch normalization is not included in the original paper.
            # Authors included it in their subsequent work [https://arxiv.org/pdf/1606.06650].
            # Here it has thus been implemented.
            NN.BatchNorm2d(outChannels),
            NN.ReLU(inplace = True),
            NN.Conv2d(outChannels, outChannels, kernel_size = 3, padding = 1),
            NN.BatchNorm2d(outChannels),
            NN.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.convolutionOperation(x)

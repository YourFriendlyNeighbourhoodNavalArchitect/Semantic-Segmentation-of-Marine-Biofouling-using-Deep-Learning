import torch.nn as NN

class AttentionGates(NN.Module):
    # Attention gates modulate the flow of information between the encoder and the decoder.
    # Refer to [https://arxiv.org/pdf/1804.03999].
    # This mechanism was not implemented in the original U-Net architecture, but offers performance improvements.
    def __init__(self, decoderInput, encoderInput, intermediateChannels):
        super(AttentionGates, self).__init__()
        # Sub-module that processes the decoder input. Kernel size of 1 does not affect spatial dimensions.
        self.decoderModule = NN.Sequential(
            NN.Conv2d(decoderInput, intermediateChannels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            NN.BatchNorm2d(intermediateChannels)
        )
        # Sub-module that processes the encoder input.
        self.encoderModule = NN.Sequential(
            NN.Conv2d(encoderInput, intermediateChannels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            NN.BatchNorm2d(intermediateChannels)
        )
        # Attention mechanism, that calculates the required attention coefficients.
        self.attentionModule = NN.Sequential(
            NN.Conv2d(intermediateChannels, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            NN.BatchNorm2d(1),
            NN.Sigmoid()
        )
        self.relu = NN.ReLU(inplace = True)
 
    def forward(self, g, x):
        g1 = self.decoderModule(g)
        x1 = self.encoderModule(x)
        attentionCoefficients = self.relu(g1 + x1)
        attentionCoefficients = self.attentionModule(attentionCoefficients)
        # Regions with high/low attention coefficients are promoted/suppressed.
        return x * attentionCoefficients

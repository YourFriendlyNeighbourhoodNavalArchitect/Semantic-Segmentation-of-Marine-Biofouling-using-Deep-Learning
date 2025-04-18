from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, Sigmoid, ReLU

class AttentionGates(Module):
    # Attention gates modulate the flow of information between the encoder and the decoder.
    # Refer to [https://arxiv.org/pdf/1804.03999].
    # This mechanism was not implemented in the original U-Net architecture, but may offer performance improvements.
    def __init__(self, decoderInput, encoderInput, intermediateChannels):
        super().__init__()
        # Sub-module that processes the decoder input. Kernel size of 1 does not affect spatial dimensions.
        self.decoderModule = Sequential(Conv2d(decoderInput, intermediateChannels, kernel_size = 1),
                                        BatchNorm2d(intermediateChannels))
        # Sub-module that processes the encoder input.
        self.encoderModule = Sequential(Conv2d(encoderInput, intermediateChannels, kernel_size = 1),
                                        BatchNorm2d(intermediateChannels))
        # Attention mechanism, that calculates the required attention coefficients.
        self.attentionModule = Sequential(Conv2d(intermediateChannels, 1, kernel_size = 1),
                                          BatchNorm2d(1),
                                          Sigmoid())
        self.relu = ReLU(inplace = True)
 
    def forward(self, g, x):
        g1 = self.decoderModule(g)
        x1 = self.encoderModule(x)
        attentionCoefficients = self.relu(g1 + x1)
        attentionCoefficients = self.attentionModule(attentionCoefficients)
        # Regions with high/low attention coefficients are promoted/suppressed.
        return x * attentionCoefficients

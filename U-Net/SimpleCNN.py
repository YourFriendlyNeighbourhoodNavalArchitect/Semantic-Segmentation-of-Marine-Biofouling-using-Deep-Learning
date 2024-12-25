import torch.nn as NN
from torch.nn.functional import relu, interpolate

class SimpleCNN(NN.Module):
    # Simple architecture utilizing successive convolution blocks.
    # Missing skip connections and encoder-decoder structure.
    # Serves purpose of comparison with Attention U-Net to evaluate accuracy results.
    def __init__(self, inChannels, numClasses):
        super().__init__()
        self.numClasses = numClasses
        
        self.convolutionOne = NN.Conv2d(inChannels, 64, kernel_size = 3, padding = 1)
        self.batchNormOne = NN.BatchNorm2d(64)

        self.convolutionTwo = NN.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.batchNormTwo = NN.BatchNorm2d(128)
        
        self.convolutionThree = NN.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.batchNormThree = NN.BatchNorm2d(256)
        
        self.fullyConnectedOne = NN.Conv2d(256, 128, kernel_size = 1)
        self.batchNormFour = NN.BatchNorm2d(128)
        
        self.fullyConnectedTwo = NN.Conv2d(128, numClasses, kernel_size = 1)
        
        self.pooling = NN.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = relu(self.batchNormOne(self.convolutionOne(x)))
        x = self.pooling(x)
        
        x = relu(self.batchNormTwo(self.convolutionTwo(x)))
        x = self.pooling(x)
        
        x = relu(self.batchNormThree(self.convolutionThree(x)))
        x = self.pooling(x)
        
        x = relu(self.batchNormFour(self.fullyConnectedOne(x)))
        x = self.fullyConnectedTwo(x)
        
        x = interpolate(x, scale_factor = 8, mode = 'bilinear', align_corners = True)
        return x
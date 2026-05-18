from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Upsample
from torch import cat

class SimpleConvBlock(Module):
    # Simple convolution block without attention mechanisms.
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convolution = Sequential(
            Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            BatchNorm2d(outChannels),
            ReLU(inplace=True),
            Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            BatchNorm2d(outChannels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convolution(x)

class SimpleDownSample(Module):
    # Simple downsampling block without attention.
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convolution = SimpleConvBlock(inChannels, outChannels)
        self.pooling = MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        down = self.convolution(x)
        pooling = self.pooling(down)
        return down, pooling

class SimpleUpSample(Module):
    # Simple upsampling block without attention gates.
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.upsampling = Sequential(
            Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2d(inChannels, inChannels // 2, kernel_size=3, padding=1),
            BatchNorm2d(inChannels // 2),
            ReLU(inplace=True)
        )
        self.convolution = SimpleConvBlock(inChannels, outChannels)
       
    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        x = cat([x1, x2], dim=1)
        return self.convolution(x)

class SimpleUNet(Module):
    # Simple U-Net architecture without any attention mechanisms.
    # Network output is of the form (B, C, H, W).
    def __init__(self, inChannels, numClasses):
        super().__init__()
        self.numClasses = numClasses

        self.downConvolutionOne = SimpleDownSample(inChannels, 64)
        self.downConvolutionTwo = SimpleDownSample(64, 128)
        self.downConvolutionThree = SimpleDownSample(128, 256)
        self.downConvolutionFour = SimpleDownSample(256, 512)

        self.bottleneck = SimpleConvBlock(512, 1024)

        self.upConvolutionOne = SimpleUpSample(1024, 512)
        self.upConvolutionTwo = SimpleUpSample(512, 256)
        self.upConvolutionThree = SimpleUpSample(256, 128)
        self.upConvolutionFour = SimpleUpSample(128, 64)

        self.output = Conv2d(64, numClasses, kernel_size=1)
        
    def forward(self, x):
        downOne, poolingOne = self.downConvolutionOne(x)
        downTwo, poolingTwo = self.downConvolutionTwo(poolingOne)
        downThree, poolingThree = self.downConvolutionThree(poolingTwo)
        downFour, poolingFour = self.downConvolutionFour(poolingThree)

        bottleneck = self.bottleneck(poolingFour)

        upOne = self.upConvolutionOne(bottleneck, downFour)
        upTwo = self.upConvolutionTwo(upOne, downThree)
        upThree = self.upConvolutionThree(upTwo, downTwo)
        upFour = self.upConvolutionFour(upThree, downOne)

        output = self.output(upFour)
        return output

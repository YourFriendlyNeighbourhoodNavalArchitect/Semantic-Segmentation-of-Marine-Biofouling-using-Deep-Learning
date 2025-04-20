from torch.nn import Module, Conv2d
from UpSample import UpSample
from DownSample import DownSample
from BlueArrow import BlueArrow

class UNet(Module):
    # Variation of the original paper implementation, consisting of 5 downsampling and 5 upsampling blocks (instead of 4).
    # Network output is of the form (B, C, H, W).
    def __init__(self, inChannels, numClasses):
        super().__init__()
        self.numClasses = numClasses

        self.downConvolutionOne = DownSample(inChannels, 64)
        self.downConvolutionTwo = DownSample(64, 128)
        self.downConvolutionThree = DownSample(128, 256)
        self.downConvolutionFour = DownSample(256, 512)
        self.downConvolutionFive = DownSample(512, 1024)

        self.bottleneck = BlueArrow(1024, 2048)

        self.upConvolutionOne = UpSample(2048, 1024)
        self.upConvolutionTwo = UpSample(1024, 512)
        self.upConvolutionThree = UpSample(512, 256)
        self.upConvolutionFour = UpSample(256, 128)
        self.upConvolutionFive = UpSample(128, 64)

        self.output = Conv2d(64, numClasses, kernel_size = 1)
        
    def forward(self, x):
        downOne, poolingOne = self.downConvolutionOne(x)
        downTwo, poolingTwo = self.downConvolutionTwo(poolingOne)
        downThree, poolingThree = self.downConvolutionThree(poolingTwo)
        downFour, poolingFour = self.downConvolutionFour(poolingThree)
        downFive, poolingFive = self.downConvolutionFive(poolingFour)

        bottleneck = self.bottleneck(poolingFive)

        upOne = self.upConvolutionOne(bottleneck, downFive)
        upTwo = self.upConvolutionTwo(upOne, downFour)
        upThree = self.upConvolutionThree(upTwo, downThree)
        upFour = self.upConvolutionFour(upThree, downTwo)
        upFive = self.upConvolutionFive(upFour, downOne)

        output = self.output(upFive)
        return output

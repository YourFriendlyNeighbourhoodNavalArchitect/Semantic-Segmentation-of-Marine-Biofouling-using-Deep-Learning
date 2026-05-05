from torch.nn import Module, Conv2d
from u_net_models.UpSample import UpSample
from u_net_models.DownSample import DownSample
from u_net_models.BlueArrow import BlueArrow

class UNet(Module):
    # Architecture largely based on the original paper.
    # Network output is of the form (B, C, H, W).
    def __init__(self, inChannels, numClasses):
        super().__init__()
        self.numClasses = numClasses

        self.downConvolutionOne = DownSample(inChannels, 64)
        self.downConvolutionTwo = DownSample(64, 128)
        self.downConvolutionThree = DownSample(128, 256)
        self.downConvolutionFour = DownSample(256, 512)

        self.bottleneck = BlueArrow(512, 1024)

        self.upConvolutionOne = UpSample(1024, 512)
        self.upConvolutionTwo = UpSample(512, 256)
        self.upConvolutionThree = UpSample(256, 128)
        self.upConvolutionFour = UpSample(128, 64)

        self.output = Conv2d(64, numClasses, kernel_size = 1)
        
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
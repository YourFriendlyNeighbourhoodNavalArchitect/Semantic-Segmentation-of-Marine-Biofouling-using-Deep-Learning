import sys

sys.path.append(r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\Training")

import torch
import torchviz
from torchsummary import summary
from UNet import UNet
from initializeWeights import initializeWeights
from trainingInitialization import setupDevice

# Dummy script to visualize the network and confirm its structure and output form.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Segmentation is performed in 4 different classes that will be further explained.
model = UNet(inChannels = 3, numClasses = 4).to(device)
model.apply(initializeWeights)
summary(model, input_size = (3, 256, 256))

dummyInput = torch.randn(1, 3, 256, 256).to(device)
output = model(dummyInput)
print(output.shape)

dot = torchviz.make_dot(output, params = dict(model.named_parameters()))
path = r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Visualizations\Architecture graph"
outputPath = dot.render(path, format = "png", view = True)
print(f"File saved to {outputPath}.")
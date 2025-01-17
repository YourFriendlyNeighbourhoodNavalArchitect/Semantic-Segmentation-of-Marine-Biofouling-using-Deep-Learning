from torch import randn
from os.path import join
from torchviz import make_dot
from torchsummary import summary
from trainingInitialization import initializeModel
from initializeWeights import initializeWeights
from trainingInitialization import setupDevice
from configurationFile import NUM_CLASSES, VISUALIZATIONS_PATH, RESOLUTION

def visualizeModel(modelFlag):
    # Dummy script to visualize the network and confirm its structure and output form.
    device = setupDevice()
    # Segmentation is performed in various different classes, as explained in configurationFile.py.
    model = initializeModel(modelFlag = modelFlag, inChannels = 3, numClasses = NUM_CLASSES, device = device)
    model.apply(initializeWeights)
    summary(model, input_size = (3, *RESOLUTION))

    dummyInput = randn(1, 3, *RESOLUTION).to(device)
    output = model(dummyInput)
    print(output.shape)
    dot = make_dot(output, params = dict(model.named_parameters()))
    
    if modelFlag:
        path = join(VISUALIZATIONS_PATH, 'U-Net Architecture Graph')
    else:
        path = join(VISUALIZATIONS_PATH, 'Simple CNN Architecture Graph')
    outputPath = dot.render(path, format = 'png', view = True)
    print(f'File saved to {outputPath}.')

visualizeModel(False)
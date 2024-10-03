import torch
import torch.onnx
import tqdm
import os
import json
import matplotlib.pyplot as plt
from ImageDataset import ImageDataset
from UNet import UNet
from Metrics import Metrics
from initializeWeights import initializeWeights
from torch import optim
from torch.utils.data import DataLoader, random_split

def getDataloaders(dataPath, batchSize, augmentationFlag, visualizationFlag):
    # Helper function that implements the dataloaders.
    trainingDataset = ImageDataset(dataPath, augmentationFlag, visualizationFlag)
    generator = torch.Generator().manual_seed(42)
    # Training and validation split drawn from literature.
    trainingDataset, validationDataset = random_split(trainingDataset, [0.8, 0.2], generator = generator)
    trainingDataloader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True)
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(dataset = validationDataset, batch_size = batchSize, shuffle = False)
    return trainingDataloader, validationDataloader

def getOptimizer(optimizerChoices, parameters, learningRate, trial):
    # Various optimizers to train the model on, evaluating performance along the way.
    if optimizerChoices == 'Adam':
        return optim.Adam(parameters, lr = learningRate)
    elif optimizerChoices == 'AdamW':
        weightDecay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
        return optim.AdamW(parameters, lr = learningRate, weight_decay = weightDecay)
    elif optimizerChoices == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 1.0)
        return optim.SGD(parameters, lr = learningRate, momentum = momentum)

def initializeModel(inChannels, numClasses, device):
    # Model shall be sent to GPU to expedite execution.
    model = UNet(inChannels = inChannels, numClasses = numClasses).to(device)
    model.apply(initializeWeights)
    return model

def trainOneEpoch(model, dataloader, optimizer, criterion, device):
    model.train()
    runningLoss = 0

    for imagePair in tqdm.tqdm(dataloader, desc = "Training"):
        # Data shall also be sent to GPU.
        image = imagePair[0].float().to(device)
        groundTruth = imagePair[1].long().to(device)
        yPredicted = model(image)
        optimizer.zero_grad()
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (C, H, W)
        loss = criterion(yPredicted, groundTruth)
        runningLoss += loss.item()
        loss.backward()
        optimizer.step()
    return runningLoss / len(dataloader)

def validateOneEpoch(model, dataloader, criterion, device):
    model.eval()
    runningLoss = 0
    diceScore = 0
    IoUScore = 0
    numClasses = model.numClasses

    with torch.no_grad():
        for imagePair in tqdm.tqdm(dataloader, desc = "Validation"):
            # Data shall also be sent to GPU.
            image = imagePair[0].float().to(device)
            groundTruth = imagePair[1].long().to(device)
            yPredicted = model(image)
            # Input tensor form: (B, C, H, W)
            # Ground truth tensor form: (C, H, W)
            loss = criterion(yPredicted, groundTruth)
            runningLoss += loss.item()
            # Metrics are calculated during validation.
            diceScore += Metrics.diceCoefficient(yPredicted, groundTruth, numClasses)
            IoUScore += Metrics.IoU(yPredicted, groundTruth, numClasses)
    return runningLoss / len(dataloader), diceScore / len(dataloader), IoUScore / len(dataloader)

def logResults(epoch, trainingLoss, validationLoss, diceScore, IoUScore):
    print(f"\nEpoch {epoch + 1} results:\n"
        f"Training loss: {trainingLoss:.4f}\n"
        f"Validation loss: {validationLoss:.4f}\n"
        f"Dice coefficient: {diceScore:.4f}\n"
        f"IoU: {IoUScore:.4f}\n")

def plotMetrics(trainingLossPlot, validationLossPlot, diceScorePlot, IoUScorePlot, epoch):
    # Initialize the figure and axes if it's the first call.
    # Otherwise reset the axes for subsequent trial.
    if epoch == 0:
        plt.ion()
        if not hasattr(plotMetrics, 'figure'):
            plotMetrics.figure, (plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour) = plt.subplots(1, 4, figsize = (24, 5))
            titles = ["Training Loss", "Validation Loss", "Dice Coefficient", "IoU Score"]
            for axis, title in zip([plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour], titles):
                axis.set_xlabel("Epoch")
                axis.set_title(title)
                axis.grid(True)
        else:
            titles = ["Training Loss", "Validation Loss", "Dice Coefficient", "IoU Score"]
            for axis, title in zip([plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour], titles):
                axis.cla()
                axis.set_xlabel("Epoch")
                axis.set_title(title)
                axis.grid(True)
    
    epochs = range(1, epoch + 2)
    plotMetrics.axisOne.plot(epochs, trainingLossPlot, color = 'blue', label = 'Training Loss')
    plotMetrics.axisTwo.plot(epochs, validationLossPlot, color = 'green', label = 'Validation Loss')
    plotMetrics.axisThree.plot(epochs, diceScorePlot, color = 'red', label = 'Dice Coefficient')
    plotMetrics.axisFour.plot(epochs, IoUScorePlot, color = 'magenta', label = 'IoU Score')

    for axis in [plotMetrics.axisOne, plotMetrics.axisTwo, plotMetrics.axisThree, plotMetrics.axisFour]:
        axis.set_xlim(1, epoch + 2)
        axis.set_xticks(epochs)

    plotMetrics.figure.canvas.draw()
    plotMetrics.figure.canvas.flush_events()

def evaluatePerformance(validationLoss, diceScore, IoUScore):
    lossWeight = 0.5
    diceWeight = 0.25
    IoUWeight = 0.25
    return lossWeight * validationLoss + diceWeight * (1 - diceScore) + IoUWeight * (1 - IoUScore)

def saveBestHyperparameters(trial, savePath):
        # Fetch optimal hyperparameters in .json format for future reference.
        path = os.path.join(savePath, 'bestHyperparameters.json')
        with open(path, 'w') as f:
            json.dump(trial.params, f, indent = 4)
        print(f"Hyperparameters for best model saved at {path}.")

def saveONNX(model, device, inputShape, savePath):
        # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
        # Exporting requires dummy input tensor.
        dummyInput = torch.randn(inputShape).to(device)
        path = os.path.join(savePath, 'bestModelOverall.onnx')
        # Constant folding improves efficiency.
        torch.onnx.export(model, dummyInput, path, export_params = True, 
                          opset_version = 12, do_constant_folding = True,
                          input_names = ['Input'], output_names = ['Output'])
        print(f"Best model saved in ONNX format at {path}.")

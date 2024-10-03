import sys

sys.path.append(r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\Dataset")
sys.path.append(r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\U-Net")

import torch
import torch.nn as NN
from ImageDataset import ImageDataset
from UNet import UNet
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

def setupDevice():
    return "cuda" if torch.cuda.is_available() else "cpu"

def initializeLossFunction():
    return NN.CrossEntropyLoss()

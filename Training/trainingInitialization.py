from random import seed as PythonSeed
from numpy.random import seed as NumpySeed
from torch import manual_seed as TorchSeed
from torch.cuda import manual_seed as CUDASeed
from torch.backends import cudnn
from torch.cuda import is_available
from torch import optim
from LossFunction import LossFunction
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MyDataset import MyDataset
from UNet import UNet
from SimpleCNN import SimpleCNN
from initializeWeights import initializeWeights
from configurationFile import TRAINING_PATH, VALIDATION_PATH

def getDataloaders(batchSize):
    trainingDataset = MyDataset(TRAINING_PATH)
    validationDataset = MyDataset(VALIDATION_PATH)
    trainingDataloader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True)
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(dataset = validationDataset, batch_size = batchSize, shuffle = False)

    return trainingDataloader, validationDataloader

def getOptimizer(parameters, learningRate):
    # Weight decay requires careful tuning when implemented alongside batch normalization [https://tinyurl.com/3kzm37tz].
    # For the purposes of this dissertation, we revert to the traditional Adam optimizer, without weight decay.
    optimizer = optim.Adam(parameters, lr = learningRate)
    # Learning rate decay routine.
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', min_lr = 1e-7, factor = 0.5)

    return optimizer, scheduler

def initializeModel(modelFlag, inChannels, numClasses, device):
    # Model shall be sent to GPU to expedite execution.
    if modelFlag:
        model = UNet(inChannels = inChannels, numClasses = numClasses).to(device)
    else:
        model = SimpleCNN(inChannels = inChannels, numClasses = numClasses).to(device)
    model.apply(initializeWeights)
    return model

def setupDevice():
    if is_available():
        device = 'cuda'
        print(f'Using GPU.')
    else:
        device = 'cpu'
        print(f'Using CPU.')
    return device

def initializeLossFunction():
    # A weighted combination of cross-entropy and Dice loss is used. 
    return LossFunction()

def setSeed(seed):
    # Set global seet to ensure reproducibility.
    PythonSeed(seed)
    NumpySeed(seed)
    TorchSeed(seed)
    CUDASeed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
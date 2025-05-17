from random import seed as PythonSeed
from numpy.random import seed as NumpySeed
from torch import manual_seed as TorchSeed
from torch.cuda import manual_seed as CUDASeed
from torch.backends import cudnn
from torch.cuda import is_available
from torch import optim
from LossFunction import LossFunction
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from MyDataset import MyDataset
from UNet import UNet
from initializeWeights import initializeWeights
from configurationFile import BATCH_SIZE, WARMUP, TRAINING_PATH, VALIDATION_PATH

def getDataloaders():
    # Only the training subset is to be augmented.
    trainingDataset = MyDataset(TRAINING_PATH, augmentationFlag = True)
    validationDataset = MyDataset(VALIDATION_PATH, augmentationFlag = False)
    trainingDataloader = DataLoader(dataset = trainingDataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, num_workers = 4)
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(dataset = validationDataset, batch_size = BATCH_SIZE, shuffle = False, pin_memory = True, num_workers = 4)
    return trainingDataloader, validationDataloader

def getOptimizer(parameters, learningRate):
    # Weight decay requires careful tuning when implemented alongside batch normalization [https://tinyurl.com/3kzm37tz].
    # For the purposes of this paper, we revert to the traditional Adam optimizer, without weight decay.
    optimizer = optim.Adam(parameters, lr = learningRate)
    # Learning rate decay routines.
    warmupScheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: (epoch + 1) / WARMUP if epoch < WARMUP else 1.0)
    mainScheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.9, min_lr = 1e-6)
    return optimizer, warmupScheduler, mainScheduler

def initializeModel(inChannels, numClasses, device):
    # Model shall be sent to GPU to expedite execution.
    model = UNet(inChannels = inChannels, numClasses = numClasses).to(device)
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

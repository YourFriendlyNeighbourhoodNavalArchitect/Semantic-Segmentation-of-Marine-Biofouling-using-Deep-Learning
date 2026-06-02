from random import seed as PythonSeed

import torch
from numpy.random import seed as NumpySeed
from torch import optim
from torch.backends import cudnn
from torch.cuda import is_available
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from Dataset.MyDataset import MyDataset
from Training.LossFunction import LossFunction
from u_net_models.initializeWeights import initializeWeights
from u_net_models.UNet import UNet
from u_net_models.SimpleUNet import SimpleUNet
from Various.configurationFile import BATCH_SIZE, TRAINING_PATH, VALIDATION_PATH, WARMUP


def getDataloaders(
    pin_memory: bool = False,
    testFlag: bool = False,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    # Only the training subset is to be augmented.
    trainingDataset = MyDataset(TRAINING_PATH, augmentationFlag=True, testFlag=testFlag)
    validationDataset = MyDataset(VALIDATION_PATH, augmentationFlag=False, testFlag=testFlag)
    trainingDataloader = DataLoader(
        dataset=trainingDataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=0,
    )
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(
        dataset=validationDataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )
    return trainingDataloader, validationDataloader


def getOptimizer(
    parameters: any,
    learningRate: float,
) -> tuple[
    optim.Optimizer,
    optim.lr_scheduler.LRScheduler,
    optim.lr_scheduler.LRScheduler,
]:
    # Weight decay requires careful tuning when implemented alongside batch normalization [https://tinyurl.com/3kzm37tz].
    # For the purposes of this paper, we revert to the traditional Adam optimizer, without weight decay.
    optimizer = optim.Adam(parameters, lr=learningRate)
    # Learning rate decay routines.
    warmupScheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / WARMUP if epoch < WARMUP else 1.0,
    )
    mainScheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, min_lr=1e-6)
    return optimizer, warmupScheduler, mainScheduler


def initializeModel(
    inChannels: int,
    numClasses: int,
    device: str,
) -> UNet:
    # Model shall be sent to GPU to expedite execution.
    model = UNet(inChannels=inChannels, numClasses=numClasses).to(device)
    model.apply(initializeWeights)
    return model

def initializeSimpleUnetModel(
    inChannels: int,
    numClasses: int,
    device: str,
) -> SimpleUNet:
    # Model shall be sent to GPU to expedite execution.
    model = SimpleUNet(inChannels=inChannels, numClasses=numClasses).to(device)
    model.apply(initializeWeights)
    return model

def setupDevice():
    if is_available():
        device = "cuda"
        print("Using GPU.")
    else:
        device = "cpu"
        print("Using CPU.")
    return device


def initializeLossFunction() -> LossFunction:
    # A weighted combination of cross-entropy and Dice loss is used.
    return LossFunction()


def setSeed(seed: int):
    # Set global seet to ensure reproducibility.
    PythonSeed(seed)
    NumpySeed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

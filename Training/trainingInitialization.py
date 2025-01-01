from torch import Generator, optim
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ImageDataset import ImageDataset
from UNet import UNet
from SimpleCNN import SimpleCNN
from initializeWeights import initializeWeights
from configurationFile import SEED

def getDataloaders(dataPath, batchSize):
    trainingDataset = ImageDataset(dataPath)
    # Ensure reproducibility, using a global seed to split the data.
    generator = Generator().manual_seed(SEED)
    # Training and validation split drawn from literature.
    trainingDataset, validationDataset = random_split(trainingDataset, [0.8, 0.2], generator = generator)
    trainingDataloader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True, pin_memory = True)
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(dataset = validationDataset, batch_size = batchSize, shuffle = False, pin_memory = True)

    return trainingDataloader, validationDataloader

def getOptimizer(parameters, learningRate):
    # Weight decay requires careful tuning when implemented alongside batch normalization [https://tinyurl.com/3kzm37tz].
    # For the purposes of this dissertation, we revert to the traditional Adam optimizer, without weight decay.
    optimizer = optim.Adam(parameters, lr = learningRate)

    # Learning rate decay routine.
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', min_lr = 1e-8)

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
        print(f"Using GPU.")
    else:
        device = 'cpu'
        print(f"Using CPU.")
    return device

def initializeLossFunction():
    # Cross entropy loss is a standard choice when it comes to multi-class segmentation.
    return CrossEntropyLoss()
from torch import Generator, optim
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from ImageDataset import ImageDataset
from UNet import UNet
from SimpleCNN import SimpleCNN
from initializeWeights import initializeWeights
from configurationFile import SEED

def getDataloaders(dataPath, batchSize):
    # Helper function that implements the dataloaders.
    trainingDataset = ImageDataset(dataPath)
    generator = Generator().manual_seed(SEED)
    # Training and validation split drawn from literature.
    trainingDataset, validationDataset = random_split(trainingDataset, [0.8, 0.2], generator = generator)
    trainingDataloader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True, pin_memory = True)
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(dataset = validationDataset, batch_size = batchSize, shuffle = False, pin_memory = True)

    return trainingDataloader, validationDataloader

def getOptimizer(optimizerChoices, parameters, learningRate, stepSize, gamma, trial):
    # Various optimizers to train the model on, evaluating performance along the way.
    if optimizerChoices == 'Adam':
        optimizer = optim.Adam(parameters, lr = learningRate)
    elif optimizerChoices == 'AdamW':
        weightDecay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
        optimizer = optim.AdamW(parameters, lr = learningRate, weight_decay = weightDecay)
    elif optimizerChoices == 'SGD':
        weightDecay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
        momentum = trial.suggest_float('momentum', 0.6, 0.9)
        optimizer = optim.SGD(parameters, lr = learningRate, weight_decay = weightDecay, momentum = momentum)
    else:
        raise ValueError("Invalid optimizer choice.")

    # Learning rate decay routine.
    scheduler = StepLR(optimizer, step_size = stepSize, gamma = gamma)

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
        device = 'cuda:0'
        print(f"Using GPU.")
    else:
        device = 'cpu'
        print(f"Using CPU.")
    return device

def initializeLossFunction():
    # Cross entropy loss is a standard choice when it comes to multi-class segmentation.
    return CrossEntropyLoss()
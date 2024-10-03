import optuna
from trainingInitialization import getDataloaders, getOptimizer, initializeModel, setupDevice, initializeLossFunction
from trainingVisualization import logResults
from trainingPreparation import trainOneEpoch, validateOneEpoch, trainingLoop
from trainingFinalization import saveBestHyperparameters, saveONNX

def optimizeHyperparameters(trial, dataPath, device):
    learningRate = trial.suggest_float('learningRate', 1e-5, 1e-2)
    batchSize = trial.suggest_categorical('batchSize', [8, 16, 32])
    epochs = trial.suggest_int('epochs', 20, 40)
    patience = trial.suggest_int('patience', 5, 10)
    optimizerChoices = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])

    criterion = initializeLossFunction()
    model = initializeModel(inChannels = 3, numClasses = 4, device = device)
    optimizer = getOptimizer(optimizerChoices, model.parameters(), learningRate, trial)
    
    augmentationFlag = True
    visualizationFlag = False
    trainingDataloader, validationDataloader = getDataloaders(dataPath, batchSize, augmentationFlag, visualizationFlag)

    return trainingLoop(model, trainingDataloader, validationDataloader, optimizer, criterion, epochs, patience, device)

def trainModel(modelSavePath, dataPath, device):
    study = optuna.create_study(directions = ['minimize', 'maximize', 'maximize'])
    study.optimize(lambda trial: optimizeHyperparameters(trial, dataPath, device), n_trials = 30)

    bestTrial = max(study.best_trials, key = lambda t: t.values[1])
    bestHyperparameters = {key: bestTrial.params[key] for key in ['learningRate', 'batchSize', 'epochs', 'optimizer']}
    
    bestModel = initializeModel(inChannels = 3, numClasses = 4, device = device)
    bestOptimizer = getOptimizer(bestHyperparameters['optimizer'], bestModel.parameters(), bestHyperparameters['learningRate'], bestTrial)
    augmentationFlag = True
    visualizationFlag = False
    trainingDataloader, validationDataloader = getDataloaders(dataPath, bestHyperparameters['batchSize'], augmentationFlag, visualizationFlag)
    
    for epoch in range(bestHyperparameters['epochs']):
        trainingLoss = trainOneEpoch(bestModel, trainingDataloader, bestOptimizer, initializeLossFunction(), device)
        validationLoss, diceScore, IoUScore = validateOneEpoch(bestModel, validationDataloader, initializeLossFunction(), device)
        logResults(epoch, trainingLoss, validationLoss, diceScore, IoUScore)
    
    inputShape = (1, 3, 256, 256)
    saveONNX(bestModel, device, inputShape, modelSavePath)
    saveBestHyperparameters(bestTrial, modelSavePath)

modelSavePath = r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS"
dataPath = r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS"
device = setupDevice()
trainModel(modelSavePath, dataPath, device)
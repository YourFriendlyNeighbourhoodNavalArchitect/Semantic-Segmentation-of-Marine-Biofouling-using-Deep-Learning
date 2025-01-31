import optuna
from trainingInitialization import getDataloaders, getOptimizer, initializeModel, setupDevice, initializeLossFunction, setSeed
from trainingPreparation import trainingLoop
from trainingFinalization import saveONNX, saveResults, deleteResiduals
from configurationFile import SEED, MODEL_PATH, NUM_CLASSES, RESOLUTION, VISUALIZATIONS_PATH

def trainModel(modelSavePath, trainingPlotSavePath, modelFlag, device, numClasses, numTrials):
    # Ensure reproducibility between runs.
    setSeed(SEED)
    # List to keep track of all the saved files for each trial.
    savedFiles = []
    # Initiate hyperparameter optimization with respect to validation loss.
    study = optuna.create_study(direction = 'minimize')

    for _ in range(numTrials):
        trial = study.ask()
        trialNumber = trial.number
        # Common hyperparameter ranges drawn from literature.
        learningRate = trial.suggest_float('learningRate', 5e-5, 5e-4)
        batchSize = trial.suggest_categorical('batchSize', [8, 16, 32])
        epochs = trial.suggest_int('epochs', 400, 500)

        criterion = initializeLossFunction()
        model = initializeModel(modelFlag = modelFlag, inChannels = 3, numClasses = numClasses, device = device)
        optimizer, scheduler = getOptimizer(model.parameters(), learningRate)

        trainingDataloader, validationDataloader = getDataloaders(batchSize)
        trainingMetrics, validationMetrics, PNGPath = trainingLoop(model, trainingDataloader, validationDataloader, optimizer, scheduler, criterion, epochs, device, trialNumber)
        
        inputShape = (1, 3, *RESOLUTION)

        # Report the results of the trial and save valuable results.
        ONNXPath = saveONNX(model, device, inputShape, modelSavePath, trialNumber)
        JSONPath = saveResults(trial, trainingMetrics, validationMetrics, modelSavePath)
        savedFiles.append((ONNXPath, JSONPath, PNGPath))
        validationLoss = validationMetrics['Loss']
        study.tell(trial, validationLoss)

    # Obtain optimal trial.
    bestTrial = study.best_trial
    
    # Clean up non-optimal saved files.
    deleteResiduals(savedFiles, bestTrial.number, modelSavePath, trainingPlotSavePath, modelFlag)

modelFlag = True
device = setupDevice()
numTrials = 1
trainModel(MODEL_PATH, VISUALIZATIONS_PATH, modelFlag, device, NUM_CLASSES, numTrials)

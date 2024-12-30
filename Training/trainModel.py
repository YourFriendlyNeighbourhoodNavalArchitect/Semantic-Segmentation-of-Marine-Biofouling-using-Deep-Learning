import optuna
from torch.backends import cudnn
from trainingInitialization import getDataloaders, getOptimizer, initializeModel, setupDevice, initializeLossFunction
from trainingPreparation import trainingLoop
from trainingFinalization import saveONNX, saveResults, deleteResiduals
from trainingVisualization import saveTrainingPlot
from configurationFile import MODEL_PATH, NUM_CLASSES, TRAINING_PATH, RESOLUTION, VISUALISATIONS_PATH

def trainModel(modelSavePath, trainingPlotSavePath, dataPath, modelFlag, device, numClasses, numTrials):
    # Ensure reproducibility between runs.
    cudnn.deterministic = True
    cudnn.benchmark = False
    # List to keep track of all the saved files for each trial.
    savedFiles = []
    # Acquiring Pareto front of optimal solutions.
    study = optuna.create_study(directions = ['minimize', 'maximize', 'maximize'])

    for _ in range(numTrials):
        trial = study.ask()
        trialNumber = trial.number
        # Common hyperparameter ranges drawn from literature.
        learningRate = trial.suggest_float('learningRate', 1e-4, 1e-2)
        batchSize = trial.suggest_categorical('batchSize', [8, 16])
        epochs = trial.suggest_int('epochs', 3, 3)
        patience = trial.suggest_int('patience', 40, 60)
        optimizerChoices = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
        stepSize = trial.suggest_int('step_size', 10, 25)
        gamma = trial.suggest_float('lr_decay_factor', 0.1, 0.9)

        criterion = initializeLossFunction()
        model = initializeModel(modelFlag = modelFlag, inChannels = 3, numClasses = numClasses, device = device)
        optimizer, scheduler = getOptimizer(optimizerChoices, model.parameters(), learningRate, stepSize, gamma, trial)

        trainingDataloader, validationDataloader = getDataloaders(dataPath, batchSize)
        trainingLoss, validationLoss, diceScore, IoUScore, accuracy, precision, recall, PNGPath = trainingLoop(model, trainingDataloader, validationDataloader, optimizer, scheduler, criterion, epochs, patience, device, trialNumber)
        
        inputShape = (1, 3, *RESOLUTION)

        # Report the results of the trial and save valuable results.
        ONNXPath = saveONNX(model, device, inputShape, modelSavePath, trialNumber)
        JSONPath = saveResults(trial, trainingLoss, validationLoss, diceScore, IoUScore, accuracy, precision, recall, modelSavePath)
        savedFiles.append((ONNXPath, JSONPath, PNGPath))
        study.tell(trial, (validationLoss, diceScore, IoUScore))

    # Obtain Pareto-optimal trial with highest Dice coefficient.
    bestTrial = max(study.best_trials, key = lambda t: t.values[1])
    
    # Clean up non-optimal saved files
    deleteResiduals(savedFiles, bestTrial.number, modelSavePath, trainingPlotSavePath, modelFlag)

modelFlag = False
device = setupDevice()
numTrials = 1
trainModel(MODEL_PATH, VISUALISATIONS_PATH, TRAINING_PATH, modelFlag, device, NUM_CLASSES, numTrials)
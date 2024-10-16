import optuna
from trainingInitialization import getDataloaders, getOptimizer, initializeModel, setupDevice, initializeLossFunction
from trainingPreparation import trainingLoop
from trainingFinalization import saveONNX, saveBestHyperparameters, deleteResiduals

def trainModel(modelSavePath, dataPath, device, numClasses, numTrials):
    # List to keep track of all the saved files for each trial.
    savedFiles = []
    # Acquiring Pareto front of optimal solutions.
    study = optuna.create_study(directions = ['minimize', 'maximize', 'maximize'])

    for _ in range(numTrials):
        trial = study.ask()
        # Common hyperparameter ranges drawn from literature.
        learningRate = trial.suggest_float('learningRate', 1e-4, 1e-2)
        batchSize = trial.suggest_categorical('batchSize', [8, 16, 32])
        epochs = trial.suggest_int('epochs', 30, 50)
        patience = trial.suggest_int('patience', 5, 10)
        optimizerChoices = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])

        criterion = initializeLossFunction()
        model = initializeModel(inChannels = 3, numClasses = numClasses, device = device)
        optimizer = getOptimizer(optimizerChoices, model.parameters(), learningRate, trial)

        augmentationFlag = True
        trainingDataloader, validationDataloader = getDataloaders(dataPath, numClasses, batchSize, augmentationFlag)
        validationLoss, diceScore, IoUScore = trainingLoop(model, trainingDataloader, validationDataloader, optimizer, criterion, epochs, patience, device)

        trialNumber = trial.number
        inputShape = (1, 3, 256, 256)

        # Save the model and record the files saved.
        ONNXPath = saveONNX(model, device, inputShape, modelSavePath, trialNumber)
        JSONPath = saveBestHyperparameters(trial, modelSavePath)
        savedFiles.append((ONNXPath, JSONPath))

        # Report the results of the trial
        study.tell(trial, (validationLoss, diceScore, IoUScore))

    # Obtain Pareto-optimal trial with highest Dice coefficient.
    bestTrial = max(study.best_trials, key = lambda t: t.values[1])
    bestHyperparameters = {key: bestTrial.params[key] for key in ['learningRate', 'batchSize', 'epochs', 'optimizer']}
    
    # Clean up non-optimal saved files
    deleteResiduals(savedFiles, bestTrial.number, modelSavePath)

    print(f"Best trial number: {bestTrial.number}")
    print(f"Best validation loss: {bestTrial.values[0]:.4f}")
    print(f"Best Dice coefficient: {bestTrial.values[1]:.4f}")
    print(f"Best IoU score: {bestTrial.values[2]:.4f}")
    print("Best hyperparameters found:")
    for key, value in bestHyperparameters.items():
        print(f"{key}: {value}")

modelSavePath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained model'
dataPath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TRAINING'
device = setupDevice()
numClasses = 4
numTrials = 50
trainModel(modelSavePath, dataPath, device, numClasses, numTrials)

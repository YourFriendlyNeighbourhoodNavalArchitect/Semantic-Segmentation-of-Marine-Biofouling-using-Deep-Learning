from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from trainingInitialization import getDataloaders, getOptimizer, initializeModel, setupDevice, initializeLossFunction, setSeed
from trainingPreparation import trainingLoop
from trainingFinalization import saveONNX, saveResults, deleteResiduals
from configurationFile import SEED, RESOLUTION, NUM_CLASSES, MODEL_PATH

def trainModel(savePath, device, numClasses, numTrials):
    # Ensure reproducibility between runs.
    setSeed(SEED)
    # List to keep track of all the saved files for each trial.
    savedFiles = []
    # Initiate hyperparameter optimization with respect to validation loss.
    # Use Tree-structured Parzen Estimator (TPE) to exploe hyperparameter space.
    study = create_study(direction = 'minimize', sampler = TPESampler(seed = SEED), 
                         pruner = MedianPruner(n_startup_trials = 5, n_warmup_steps = 20))

    def objective(trial):
        learningRate = trial.suggest_float('learningRate', 1e-5, 1e-3, log = True)
        criterion = initializeLossFunction()
        model = initializeModel(inChannels = 3, numClasses = numClasses, device = device)
        optimizer, warmupScheduler, mainScheduler = getOptimizer(model.parameters(), learningRate)
        trainingDataloader, validationDataloader = getDataloaders()
        trainingMetrics, validationMetrics, PNGPath, maxEpochs = trainingLoop(model, trial, trainingDataloader, validationDataloader, optimizer,
                                                                              warmupScheduler, mainScheduler, criterion, device)
        inputShape = (1, 3, *RESOLUTION)
        # Save valuable trial results separately.
        ONNXPath = saveONNX(model, device, inputShape, MODEL_PATH, trial.number)
        JSONPath = saveResults(trial, maxEpochs, trainingMetrics, validationMetrics, MODEL_PATH)
        savedFiles.append((ONNXPath, JSONPath, PNGPath))
        return validationMetrics['Loss']

    # Obtain optimal trial.
    study.optimize(objective, n_trials = numTrials)
    bestTrial = study.best_trial
    # Clean up non-optimal saved files.
    deleteResiduals(savedFiles, bestTrial.number, savePath)

if __name__ == '__main__':
    # Multiprocessing guard.
    device = setupDevice()
    numTrials = 20
    trainModel(MODEL_PATH, device, NUM_CLASSES, numTrials)
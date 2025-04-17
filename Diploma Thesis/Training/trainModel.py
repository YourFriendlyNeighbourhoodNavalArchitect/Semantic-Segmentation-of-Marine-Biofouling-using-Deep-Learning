from optuna import create_study
from optuna.samplers import GridSampler
from numpy import linspace
from trainingInitialization import getDataloaders, getOptimizer, initializeModel, setupDevice, initializeLossFunction, setSeed
from trainingPreparation import trainingLoop
from trainingFinalization import saveONNX, saveResults, deleteResiduals
from configurationFile import SEED, RESOLUTION, NUM_CLASSES, MODEL_PATH, VISUALIZATIONS_PATH

def trainModel(modelSavePath, trainingPlotSavePath, modelFlag, device, numClasses, numTrials):
    # Ensure reproducibility between runs.
    setSeed(SEED)
    # List to keep track of all the saved files for each trial.
    savedFiles = []
    # Initiate hyperparameter optimization with respect to validation loss.
    # Common learning rate range found in the relevant literature.
    learningRateGrid = linspace(5e-5, 5e-4, num = numTrials).tolist()
    searchSpace = {'learningRate': learningRateGrid}
    study = create_study(direction = 'minimize', sampler = GridSampler(searchSpace, seed = SEED))

    def objective(trial):
        learningRate = trial.suggest_categorical('learningRate', learningRateGrid)
        criterion = initializeLossFunction()
        model = initializeModel(modelFlag = modelFlag, inChannels = 3, numClasses = numClasses, device = device)
        optimizer, warmupScheduler, mainScheduler = getOptimizer(model.parameters(), learningRate)
        trainingDataloader, validationDataloader = getDataloaders()
        trainingMetrics, validationMetrics, PNGPath, maxEpochs = trainingLoop(model, trainingDataloader, validationDataloader, optimizer,
                                                                              warmupScheduler, mainScheduler, criterion, device, trial.number)
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
    deleteResiduals(savedFiles, bestTrial.number, modelSavePath, trainingPlotSavePath, modelFlag)

if __name__ == '__main__':
    # Multiprocessing guard.
    modelFlag = True
    device = setupDevice()
    numTrials = 25
    trainModel(MODEL_PATH, VISUALIZATIONS_PATH, modelFlag, device, NUM_CLASSES, numTrials)

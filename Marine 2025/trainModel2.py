from matplotlib.path import Path
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from Training.trainingFinalization import deleteResiduals, saveONNX, saveResults
from Training.trainingInitialization import (
    getDataloaders,
    getOptimizer,
    initializeLossFunction,
    setSeed,
    setupDevice,
)
from Training.trainingPreparation import trainingLoop
from u_net_models.SimpleUNet import SimpleUNet
from Various.configurationFile import MODEL_PATH, RESOLUTION, SEED, NUM_CLASSES_new


def trainModel2(savePath: Path, device: str, numClasses: int, numTrials: int):
    # Ensure reproducibility between runs.
    setSeed(SEED)
    # List to keep track of all the saved files for each trial.
    savedFiles = []
    # Initiate hyperparameter optimization with respect to validation loss.
    # Use Tree-structured Parzen Estimator (TPE) to explore hyperparameter space.
    study = create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=30),
    )

    def objective(trial):
        learningRate = trial.suggest_float("learningRate", 1e-5, 1e-3, log=True)
        criterion = initializeLossFunction()
        # Use SimpleUNet instead of attention-based UNet
        model = SimpleUNet(inChannels=3, numClasses=numClasses).to(device)
        optimizer, warmupScheduler, mainScheduler = getOptimizer(
            model.parameters(), learningRate
        )
        trainingDataloader, validationDataloader = getDataloaders()
        trainingMetrics, validationMetrics, PNGPath, maxEpochs = trainingLoop(
            model,
            trial,
            trainingDataloader,
            validationDataloader,
            optimizer,
            warmupScheduler,
            mainScheduler,
            criterion,
            device,
        )
        inputShape = (1, 3, *RESOLUTION)
        # Save valuable trial results separately.
        ONNXPath = saveONNX(model, device, inputShape, MODEL_PATH, trial.number)
        JSONPath = saveResults(
            trial, maxEpochs, trainingMetrics, validationMetrics, MODEL_PATH
        )
        savedFiles.append((ONNXPath, JSONPath, PNGPath))
        return validationMetrics["Loss"]

    # Obtain optimal trial.
    study.optimize(objective, n_trials=numTrials)
    bestTrial = study.best_trial
    # Clean up non-optimal saved files.
    deleteResiduals(savedFiles, bestTrial.number, savePath)


if __name__ == "__main__":
    # Multiprocessing guard.
    device = setupDevice()
    numTrials = 50
    trainModel2(MODEL_PATH, device, NUM_CLASSES_new, numTrials)

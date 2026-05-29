"""
Train a Simple U-Net (no attention mechanisms) as a baseline comparison.
Uses the same hyperparameters as the best Attention U-Net trial for a fair ablation.
No Optuna — single deterministic run.
"""

from Training.trainingInitialization import (
    getDataloaders,
    getOptimizer,
    initializeLossFunction,
    setSeed,
    setupDevice,
)
from Training.trainingPreparation import trainingLoop
from Training.trainingFinalization import (
    saveONNX,
    saveCheckpoint,
    saveResults,
    cleanupAndRename,
)
from u_net_models.SimpleUNet import SimpleUNet
from u_net_models.initializeWeights import initializeWeights
from Various.configurationFile import (
    MODEL_PATH,
    NUM_CLASSES,
    RESOLUTION,
    SEED,
    LEARNING_RATE,
)


def trainSimpleUNet():
    # Reproducibility.
    setSeed(SEED)
    device = setupDevice()

    # Output directory for the Simple U-Net (separate from the Attention U-Net).
    savePath = MODEL_PATH / 'SimpleUNet'
    savePath.mkdir(exist_ok = True, parents = True)

    # Model initialisation with Kaiming weights (same as Attention U-Net).
    model = SimpleUNet(inChannels = 3, numClasses = NUM_CLASSES).to(device)
    model.apply(initializeWeights)

    # Same training setup as the best Attention U-Net trial.
    criterion = initializeLossFunction()
    optimizer, warmupScheduler, mainScheduler = getOptimizer(model.parameters(), LEARNING_RATE)
    trainingDataloader, validationDataloader = getDataloaders()

    # Train.
    trialNumber = 0
    trainingMetrics, validationMetrics, PNGPath, maxEpochs = trainingLoop(
        model, trainingDataloader, validationDataloader, optimizer,
        warmupScheduler, mainScheduler, criterion, device, trialNumber = trialNumber
    )

    # Save all artifacts.
    inputShape = (1, 3, *RESOLUTION)
    saveONNX(model, device, inputShape, savePath, trialNumber)
    saveCheckpoint(model, optimizer, maxEpochs, validationMetrics, savePath, trialNumber)
    saveResults(maxEpochs, LEARNING_RATE, trainingMetrics, validationMetrics, savePath, trialNumber)
    cleanupAndRename(trialNumber, savePath)

    print(f'\nTraining complete after {maxEpochs} epochs.')
    print(f'Validation Dice: {validationMetrics["Dice Coefficient"]:.4f}')
    print(f'Validation IoU:  {validationMetrics["IoU"]:.4f}')


if __name__ == '__main__':
    trainSimpleUNet()
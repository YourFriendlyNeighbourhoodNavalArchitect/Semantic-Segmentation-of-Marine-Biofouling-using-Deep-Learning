"""Train a Simple U-Net (no attention mechanisms) as a baseline comparison. Uses the same hyperparameters as the best Attention U-Net trial for a fair ablation."""

from Training.trainingFinalization import (
    cleanupAndRename,
    saveCheckpoint,
    saveONNX,
    saveResults,
)
from Training.trainingInitialization import (
    getDataloaders,
    getOptimizer,
    initializeLossFunction,
    initializeSimpleUnetModel,
    setSeed,
    setupDevice,
)
from Training.trainingPreparation import trainingLoop
from Various.configurationFile import (
    LEARNING_RATE,
    MODEL_PATH,
    RESOLUTION,
    SEED,
    VISUALIZATIONS_PATH,
    NUM_CLASSES_new,
)


def trainSimpleUNet():
    # Reproducibility.
    setSeed(SEED)
    device = setupDevice()

    savePath = MODEL_PATH / "SimpleUNet"
    plotPath = VISUALIZATIONS_PATH / "SimpleUNet"
    savePath.mkdir(exist_ok=True, parents=True)
    plotPath.mkdir(exist_ok=True, parents=True)

    # Model initialisation with Kaiming weights (same as Attention U-Net).

    model = initializeSimpleUnetModel(inChannels=3, numClasses=NUM_CLASSES_new, device=device)
    # Same training setup as the best Attention U-Net trial.
    criterion = initializeLossFunction()
    optimizer, warmupScheduler, mainScheduler = getOptimizer(
        parameters=model.parameters(),
        learningRate=LEARNING_RATE,
    )
    trainingDataloader, validationDataloader = getDataloaders(testFlag=True)

    # Train.
    trialNumber = 0
    trainingMetrics, validationMetrics, maxEpochs = trainingLoop(
        model,
        trainingDataloader,
        validationDataloader,
        optimizer,
        warmupScheduler,
        mainScheduler,
        criterion,
        device,
        plotPath=plotPath,
        savePath=savePath,
        trialNumber=trialNumber,
    )

    # Save all artifacts.
    inputShape = (1, 3, *RESOLUTION)
    saveONNX(model, device, inputShape, savePath, trialNumber)
    saveCheckpoint(
        model, optimizer, maxEpochs, validationMetrics, savePath, trialNumber,
    )
    saveResults(
        maxEpochs,
        LEARNING_RATE,
        trainingMetrics,
        validationMetrics,
        savePath,
        trialNumber,
    )
    # cleanupAndRename(trialNumber, savePath)

    print(f"\nSimple U-Net training complete after {maxEpochs} epochs.")
    print(f"Validation Dice: {validationMetrics['Dice Coefficient']:.4f}")
    print(f"Validation IoU:  {validationMetrics['IoU']:.4f}")


if __name__ == "__main__":
    trainSimpleUNet()

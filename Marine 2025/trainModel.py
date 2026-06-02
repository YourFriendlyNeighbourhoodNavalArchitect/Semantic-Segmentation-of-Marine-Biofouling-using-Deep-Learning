"""Train an Attention U-Net."""

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
    initializeModel,
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


def trainAttentionUNet():
    # Reproducibility.
    setSeed(SEED)
    device = setupDevice()

    savePath = MODEL_PATH / "AttentionUNet"
    plotPath = VISUALIZATIONS_PATH / "AttentionUNet"
    savePath.mkdir(exist_ok=True, parents=True)
    plotPath.mkdir(exist_ok=True, parents=True)

    # Attention U-Net: same 64→128→256→512→1024 structure,
    # with spatial attention gates + squeeze-and-excitation blocks.
    model = initializeModel(inChannels=3, numClasses=NUM_CLASSES_new, device=device)

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

    print(f"\nAttention U-Net training complete after {maxEpochs} epochs.")
    print(f"Validation Dice: {validationMetrics['Dice Coefficient']:.4f}")
    print(f"Validation IoU:  {validationMetrics['IoU']:.4f}")


if __name__ == "__main__":
    trainAttentionUNet()

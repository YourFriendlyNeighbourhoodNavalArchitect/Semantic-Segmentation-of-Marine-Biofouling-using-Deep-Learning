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
    NUM_CLASSES_v2,
)


def trainAttentionUNet():
    setSeed(SEED)
    device = setupDevice()

    savePath = MODEL_PATH / "AttentionUNet"
    savePath.mkdir(exist_ok=True, parents=True)

    # Attention U-Net: same 64→128→256→512→1024 structure,
    # with spatial attention gates + squeeze-and-excitation blocks.
    model = initializeModel(inChannels=3, numClasses=NUM_CLASSES_v2, device=device)

    criterion = initializeLossFunction()
    optimizer, warmupScheduler, mainScheduler = getOptimizer(
        parameters=model.parameters(),
        learningRate=LEARNING_RATE,
    )
    trainingDataloader, validationDataloader = getDataloaders()

    trialNumber = 0
    trainingMetrics, validationMetrics, _, maxEpochs = trainingLoop(
        model,
        trainingDataloader,
        validationDataloader,
        optimizer,
        warmupScheduler,
        mainScheduler,
        criterion,
        device,
        trialNumber=trialNumber,
    )

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
    cleanupAndRename(trialNumber, savePath)

    print(f"\nAttention U-Net training complete after {maxEpochs} epochs.")
    print(f"Validation Dice: {validationMetrics['Dice Coefficient']:.4f}")
    print(f"Validation IoU:  {validationMetrics['IoU']:.4f}")


if __name__ == "__main__":
    trainAttentionUNet()

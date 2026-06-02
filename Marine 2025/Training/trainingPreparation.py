from pathlib import Path
from xml.parsers.expat import model

from torch import no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Training.computeMetrics import computeMetrics
from Training.LossFunction import LossFunction
from Training.trainingFinalization import saveTrialData
from Training.trainingVisualization import (
    TrainingVisualization,
)
from u_net_models.UNet import UNet
from Various.configurationFile import LOG_INTERVAL, PATIENCE, WARMUP

# Keys that hold scalar (macro-averaged) metrics.
SCALAR_KEYS = ["Loss", "Dice Coefficient", "IoU", "Accuracy", "Precision", "Recall"]
# Keys that hold per-class lists.
PER_CLASS_KEYS = [
    "Per Class Dice",
    "Per Class IoU",
    "Per Class Accuracy",
    "Per Class Precision",
    "Per Class Recall",
]


def _emptyAggregated() -> dict[
    str,
    float | list[float] | None,
]:
    """Return a fresh aggregation dict for one epoch."""
    agg = dict.fromkeys(SCALAR_KEYS, 0.0)
    agg.update(dict.fromkeys(PER_CLASS_KEYS))
    return agg


def _accumulatePerClass(
    aggregated: dict[str, float | list[float] | None],
    batchMetrics: dict[str, float | list[float] | None],
) -> dict[str, float | list[float] | None]:
    """Element-wise accumulation of per-class metric lists."""
    for key in PER_CLASS_KEYS:
        if key in batchMetrics and batchMetrics[key] is not None:
            batchValues = batchMetrics[key]
            if aggregated[key] is None:
                aggregated[key] = [0.0] * len(batchValues)
            for i, v in enumerate(batchValues):
                aggregated[key][i] += v
    return aggregated  # Note: this modifies the input dict in-place for efficiency, but also returns it for convenience readability.


def _averageMetrics(
    aggregated: dict[str, float | list[float] | None],
    numBatches: int,
) -> dict[str, float | list[float] | None]:
    """Divide all accumulated metrics by the number of batches."""
    averaged = {}
    for key in SCALAR_KEYS:
        averaged[key] = aggregated[key] / numBatches
    for key in PER_CLASS_KEYS:
        if aggregated[key] is not None:
            averaged[key] = [v / numBatches for v in aggregated[key]]
        else:
            averaged[key] = None
    return averaged


def trainOneEpoch(
    model: UNet,
    trainingDataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: LossFunction,
    device: str,
) -> dict[str, float]:
    model.train()
    aggregated = _emptyAggregated()

    for data in tqdm(trainingDataloader, desc="Training"):
        # Send data to GPU.
        image = data[0].to(device)
        groundTruth = data[1].to(device)
        optimizer.zero_grad()
        prediction = model(image)
        loss = criterion(prediction, groundTruth)
        aggregated["Loss"] += loss.item()
        loss.backward()
        optimizer.step()
        # Compute metrics.
        batchMetrics = computeMetrics(prediction, groundTruth)
        for key in SCALAR_KEYS:
            if key != "Loss" and key in batchMetrics:
                aggregated[key] += batchMetrics[key]
        _accumulatePerClass(aggregated, batchMetrics)

    return _averageMetrics(aggregated, len(trainingDataloader))


def validateOneEpoch(
    model: UNet,
    validationDataloader: DataLoader,
    criterion: LossFunction,
    device: str,
) -> dict[str, float]:
    model.eval()
    aggregated = _emptyAggregated()

    with no_grad():
        for data in tqdm(validationDataloader, desc="Validation"):
            image = data[0].to(device)
            groundTruth = data[1].to(device)
            prediction = model(image)
            loss = criterion(prediction, groundTruth)
            aggregated["Loss"] += loss.item()
            batchMetrics = computeMetrics(prediction, groundTruth)
            for key in SCALAR_KEYS:
                if key != "Loss" and key in batchMetrics:
                    aggregated[key] += batchMetrics[key]
            _accumulatePerClass(aggregated, batchMetrics)

    return _averageMetrics(aggregated, len(validationDataloader))


def trainingLoop(
    model: UNet,
    trainingDataloader: DataLoader,
    validationDataloader: DataLoader,
    optimizer: optim.Optimizer,
    warmupScheduler: optim.lr_scheduler.LambdaLR,
    mainScheduler: optim.lr_scheduler.LambdaLR,
    criterion: LossFunction,
    device: str,
    plotPath: Path,
    savePath: Path,
    trialNumber: int = 0,
) -> tuple[dict[str, float], dict[str, float], int]:
    trainingLossPlot = []
    validationLossPlot = []
    validationDiceScorePlot = []
    validationIoUScorePlot = []
    # Early stopping mechanism.
    bestValidationLoss = float("inf")
    patienceCounter = 0
    epochs = 0
    trainVizualizationObject = TrainingVisualization(plotPath=plotPath)
    maxEpochs = 300
    while True:
        trainingMetrics = trainOneEpoch(
            model=model,
            trainingDataloader=trainingDataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        validationMetrics = validateOneEpoch(
            model=model,
            validationDataloader=validationDataloader,
            criterion=criterion,
            device=device,
        )
        currentLR = optimizer.param_groups[0]["lr"]
        epochs += 1
        if epochs < WARMUP:
            warmupScheduler.step()
        else:
            mainScheduler.step(validationMetrics["Loss"])

        # Append to plot lists every epoch (for smooth curves).
        trainingLossPlot.append(trainingMetrics["Loss"])
        validationLossPlot.append(validationMetrics["Loss"])
        validationDiceScorePlot.append(validationMetrics["Dice Coefficient"])
        validationIoUScorePlot.append(validationMetrics["IoU"])

        # Log and save every LOG_INTERVAL epochs, and always on the first epoch.
        if epochs == 1 or epochs % LOG_INTERVAL == 0:
            trainVizualizationObject.logResults(epochs, currentLR, trainingMetrics, validationMetrics)
            saveTrialData(
                epoch=epochs,
                currentLR=currentLR,
                trainingMetrics=trainingMetrics,
                validationMetrics=validationMetrics,
                trialNumber=trialNumber,
                savePath=savePath,
            )

        # Models train indefinitely, until validation loss stops improving.
        if validationMetrics["Loss"] < bestValidationLoss:
            bestValidationLoss = validationMetrics["Loss"]
            patienceCounter = 0
        else:
            patienceCounter += 1
        if patienceCounter >= PATIENCE or epochs >= maxEpochs:  # Safety cap to prevent infinite loops in case of bugs.
            print(f"Early stopping triggered after {epochs} epochs.")
            # Always log the final epoch.
            if epochs % LOG_INTERVAL != 0:
                trainVizualizationObject.logResults(epochs, currentLR, trainingMetrics, validationMetrics)
                saveTrialData(
                    epoch=epochs,
                    currentLR=currentLR,
                    trainingMetrics=trainingMetrics,
                    validationMetrics=validationMetrics,
                    trialNumber=trialNumber,
                    savePath=savePath,
                )
            break
    # Plot training metrics after training ends.
    trainVizualizationObject.plotMetrics(
        trainingLossPlot,
        validationLossPlot,
        validationDiceScorePlot,
        validationIoUScorePlot,
        trialNumber,
    )
    return trainingMetrics, validationMetrics, epochs

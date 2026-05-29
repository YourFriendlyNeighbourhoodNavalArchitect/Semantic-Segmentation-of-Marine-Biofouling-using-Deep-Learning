from tqdm import tqdm
from torch import no_grad, cat, argmax
from Training.trainingVisualization import logResults, plotMetrics
from Training.trainingFinalization import saveTrialData
from Training.computeMetrics import computeMetrics
from Various.configurationFile import WARMUP, PATIENCE, LOG_INTERVAL


# Keys that hold scalar (macro-averaged) metrics.
SCALAR_KEYS = ['Loss', 'Dice Coefficient', 'IoU', 'Accuracy', 'Precision', 'Recall']
# Keys that hold per-class lists.
PER_CLASS_KEYS = ['Per Class IoU', 'Per Class Accuracy', 'Per Class Precision', 'Per Class Recall']


def _emptyAggregated():
    """Return a fresh aggregation dict for one epoch."""
    agg = {key: 0.0 for key in SCALAR_KEYS}
    agg.update({key: None for key in PER_CLASS_KEYS})
    return agg


def _accumulatePerClass(aggregated, batchMetrics):
    """Element-wise accumulation of per-class metric lists."""
    for key in PER_CLASS_KEYS:
        if key in batchMetrics and batchMetrics[key] is not None:
            batchValues = batchMetrics[key]
            if aggregated[key] is None:
                aggregated[key] = [0.0] * len(batchValues)
            for i, v in enumerate(batchValues):
                aggregated[key][i] += v


def _averageMetrics(aggregated, numBatches):
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


def trainOneEpoch(model, trainingDataloader, optimizer, criterion, device):
    model.train()
    aggregated = _emptyAggregated()

    for data in tqdm(trainingDataloader, desc = 'Training'):
        # Send data to GPU.
        image = data[0].to(device)
        groundTruth = data[1].to(device)
        optimizer.zero_grad()
        prediction = model(image)
        loss = criterion(prediction, groundTruth)
        aggregated['Loss'] += loss.item()
        loss.backward()
        optimizer.step()
        # Compute metrics.
        batchMetrics = computeMetrics(prediction, groundTruth)
        for key in SCALAR_KEYS:
            if key != 'Loss' and key in batchMetrics:
                aggregated[key] += batchMetrics[key]
        _accumulatePerClass(aggregated, batchMetrics)

    return _averageMetrics(aggregated, len(trainingDataloader))


def validateOneEpoch(model, validationDataloader, criterion, device):
    model.eval()
    aggregated = _emptyAggregated()

    with no_grad():
        for data in tqdm(validationDataloader, desc = 'Validation'):
            image = data[0].to(device)
            groundTruth = data[1].to(device)
            prediction = model(image)
            loss = criterion(prediction, groundTruth)
            aggregated['Loss'] += loss.item()
            batchMetrics = computeMetrics(prediction, groundTruth)
            for key in SCALAR_KEYS:
                if key != 'Loss' and key in batchMetrics:
                    aggregated[key] += batchMetrics[key]
            _accumulatePerClass(aggregated, batchMetrics)

    return _averageMetrics(aggregated, len(validationDataloader))


def trainingLoop(model, trainingDataloader, validationDataloader, optimizer, warmupScheduler, mainScheduler, criterion, device, trialNumber = 0):
    trainingLossPlot = []
    validationLossPlot = []
    validationDiceScorePlot = []
    validationIoUScorePlot = []
    # Early stopping mechanism.
    bestValidationLoss = float('inf')
    patienceCounter = 0
    maxEpochs = 0

    while True:
        logged = False
        trainingMetrics = trainOneEpoch(model, trainingDataloader, optimizer, criterion, device)
        validationMetrics = validateOneEpoch(model, validationDataloader, criterion, device)
        currentLR = optimizer.param_groups[0]['lr']
        maxEpochs += 1
        if maxEpochs < WARMUP:
            warmupScheduler.step()
        else:
            mainScheduler.step(validationMetrics['Loss'])

        # Append to plot lists every epoch (for smooth curves).
        trainingLossPlot.append(trainingMetrics['Loss'])
        validationLossPlot.append(validationMetrics['Loss'])
        validationDiceScorePlot.append(validationMetrics['Dice Coefficient'])
        validationIoUScorePlot.append(validationMetrics['IoU'])

        # Log and save every LOG_INTERVAL epochs, and always on the first epoch.
        if maxEpochs == 1 or maxEpochs % LOG_INTERVAL == 0:
            logResults(maxEpochs, currentLR, trainingMetrics, validationMetrics)
            saveTrialData(maxEpochs, currentLR, trainingMetrics, validationMetrics, trialNumber)
            logged = True

        # Models train indefinitely, until validation loss stops improving.
        if validationMetrics['Loss'] < bestValidationLoss:
            bestValidationLoss = validationMetrics['Loss']
            patienceCounter = 0
        else:
            patienceCounter += 1
        if patienceCounter >= PATIENCE and not logged:
            print(f'Early stopping triggered after {maxEpochs} epochs.')
            # Always log the final epoch.
            if maxEpochs % LOG_INTERVAL != 0:
                logResults(maxEpochs, currentLR, trainingMetrics, validationMetrics)
                saveTrialData(maxEpochs, currentLR, trainingMetrics, validationMetrics, trialNumber)
            break
    
    # Plot training metrics after training ends.
    PNGPath = plotMetrics(trainingLossPlot, validationLossPlot, validationDiceScorePlot, validationIoUScorePlot, trialNumber)
    return trainingMetrics, validationMetrics, PNGPath, maxEpochs
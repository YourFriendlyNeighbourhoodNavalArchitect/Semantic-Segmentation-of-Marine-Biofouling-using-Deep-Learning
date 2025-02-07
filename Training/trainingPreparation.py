from tqdm import tqdm
from torch import no_grad
from trainingVisualization import logResults, plotMetrics
from trainingFinalization import saveTrialData
from computeMetrics import computeMetrics
from configurationFile import WARMUP, PATIENCE

def trainOneEpoch(model, trainingDataloader, optimizer, criterion, device):
    model.train()
    aggregatedMetrics = {'Loss': 0, 'Dice Coefficient': 0, 'IoU': 0, 'Accuracy': 0, 'Precision': 0, 'Recall': 0}

    for data in tqdm(trainingDataloader, desc = 'Training'):
        # Send data to GPU.
        image = data[0].to(device)
        groundTruth = data[1].to(device)
        optimizer.zero_grad()
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (B, H, W)
        prediction = model(image)
        loss = criterion(prediction, groundTruth)
        aggregatedMetrics['Loss'] += loss.item()
        loss.backward()
        optimizer.step()
        # Compute metrics.
        batchMetrics = computeMetrics(prediction, groundTruth)
        for key in batchMetrics:
            aggregatedMetrics[key] += batchMetrics[key]

    averagedMetrics = {key: value / len(trainingDataloader) for key, value in aggregatedMetrics.items()}
    return averagedMetrics

def validateOneEpoch(model, validationDataloader, criterion, device):
    model.eval()
    aggregatedMetrics = {'Loss': 0, 'Dice Coefficient': 0, 'IoU': 0, 'Accuracy': 0, 'Precision': 0, 'Recall': 0}

    with no_grad():
        for data in tqdm(validationDataloader, desc = 'Validation'):
            image = data[0].to(device)
            groundTruth = data[1].to(device)
            prediction = model(image)
            loss = criterion(prediction, groundTruth)
            aggregatedMetrics['Loss'] += loss.item()
            batchMetrics = computeMetrics(prediction, groundTruth)
            for key in batchMetrics:
                aggregatedMetrics[key] += batchMetrics[key]

    averagedMetrics = {key: value / len(validationDataloader) for key, value in aggregatedMetrics.items()}
    return averagedMetrics

def trainingLoop(model, trainingDataloader, validationDataloader, optimizer, warmupScheduler, mainScheduler, criterion, device, trialNumber):
    trainingLossPlot = []
    validationLossPlot = []
    validationDiceScorePlot = []
    validationIoUScorePlot = []
    # Early stopping mechanism.
    bestValidationLoss = float('inf')
    patienceCounter = 0
    maxEpochs = 0

    while True:
        trainingMetrics = trainOneEpoch(model, trainingDataloader, optimizer, criterion, device)
        validationMetrics = validateOneEpoch(model, validationDataloader, criterion, device)
        currentLR = optimizer.param_groups[0]['lr']
        maxEpochs += 1
        if maxEpochs < WARMUP:
            warmupScheduler.step()
        else:
            mainScheduler.step(validationMetrics['Loss'])

        # Logging of metrics.
        trainingLossPlot.append(trainingMetrics['Loss'])
        validationLossPlot.append(validationMetrics['Loss'])
        validationDiceScorePlot.append(validationMetrics['Dice Coefficient'])
        validationIoUScorePlot.append(validationMetrics['IoU'])
        logResults(maxEpochs, currentLR, trainingMetrics, validationMetrics)
        saveTrialData(maxEpochs, currentLR, trainingMetrics, validationMetrics, trialNumber)

        # Models train indefinitely, until validation loss stops improving.
        if validationMetrics['Loss'] < bestValidationLoss:
            bestValidationLoss = validationMetrics['Loss']
            patienceCounter = 0
        else:
            patienceCounter += 1
        if patienceCounter >= PATIENCE:
            print(f'Early stopping triggered after {maxEpochs} epochs.')
            break
    
    # Plot training metrics after training ends, to decrease computational overhead.
    PNGPath = plotMetrics(trainingLossPlot, validationLossPlot, validationDiceScorePlot, validationIoUScorePlot, trialNumber)
    return trainingMetrics, validationMetrics, PNGPath, maxEpochs
from tqdm import tqdm
from torch import no_grad, Tensor
from trainingVisualization import logResults, plotMetrics
from computeMetrics import computeMetrics
from configurationFile import WARMUP, PATIENCE

def trainOneEpoch(model, trainingDataloader, optimizer, criterion, device):
    model.train()
    aggregatedMetrics = {'Loss': 0, 'Dice Coefficient': 0, 'IoU': 0, 'Accuracy': 0, 'Precision': 0, 'Recall': 0}

    for data in tqdm(trainingDataloader, desc = 'Training'):
        # Send data to GPU.
        image = data[0].to(device)
        groundTruth = data[1].to(device)
        prediction = model(image)
        optimizer.zero_grad()
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (B, H, W)
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
    validationDiceScorePlot = [value.cpu().item() for value in validationDiceScorePlot]
    validationIoUScorePlot = [value.cpu().item() for value in validationIoUScorePlot]
    PNGPath = plotMetrics(trainingLossPlot, validationLossPlot, validationDiceScorePlot, validationIoUScorePlot, trialNumber)

    trainingMetrics = {key: value.cpu().item() if isinstance(value, Tensor) else value for key, value in trainingMetrics.items()}
    validationMetrics = {key: value.cpu().item() if isinstance(value, Tensor) else value for key, value in validationMetrics.items()}
    return trainingMetrics, validationMetrics, PNGPath, maxEpochs
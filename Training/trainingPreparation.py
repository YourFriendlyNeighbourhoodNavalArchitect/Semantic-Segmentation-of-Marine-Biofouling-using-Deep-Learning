from tqdm import tqdm
from torch import no_grad, Tensor
from trainingVisualization import logResults, plotMetrics
from Metrics import Metrics

def trainOneEpoch(model, trainingDataloader, optimizer, criterion, device, metricsCalculator):
    model.train()
    aggregatedMetrics = {
        'Loss': 0,
        'Dice Coefficient': 0,
        'IoU': 0,
        'Accuracy': 0,
        'Precision': 0,
        'Recall': 0}

    for imagePair in tqdm(trainingDataloader, desc = "Training"):
        # Send data to GPU.
        image = imagePair[0].to(device)
        groundTruth = imagePair[1].to(device)
        prediction = model(image)
        optimizer.zero_grad()
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (B, H, W)
        loss = criterion(prediction, groundTruth)

        # Compute metrics.
        batchMetrics = metricsCalculator.computeMetrics(prediction, groundTruth)
        aggregatedMetrics['Loss'] += loss.item()
        for key in batchMetrics:
            aggregatedMetrics[key] += batchMetrics[key]

        loss.backward()
        optimizer.step()

    averagedMetrics = {key: value / len(trainingDataloader) for key, value in aggregatedMetrics.items()}

    return averagedMetrics

def validateOneEpoch(model, validationDataloader, criterion, device, metricsCalculator):
    model.eval()
    aggregatedMetrics = {
        'Loss': 0,
        'Dice Coefficient': 0,
        'IoU': 0,
        'Accuracy': 0,
        'Precision': 0,
        'Recall': 0}

    with no_grad():
        for imagePair in tqdm(validationDataloader, desc = "Validation"):
            image = imagePair[0].to(device)
            groundTruth = imagePair[1].to(device)
            prediction = model(image)
            loss = criterion(prediction, groundTruth)

            batchMetrics = metricsCalculator.computeMetrics(prediction, groundTruth)
            aggregatedMetrics['Loss'] += loss.item()
            for key in batchMetrics:
                aggregatedMetrics[key] += batchMetrics[key]

    averagedMetrics = {key: value / len(validationDataloader) for key, value in aggregatedMetrics.items()}

    return averagedMetrics

def trainingLoop(model, trainingDataloader, validationDataloader, optimizer, scheduler, criterion, epochs, device, trialNumber):
    trainingLossPlot = []
    validationLossPlot = []
    validationDiceScorePlot = []
    validationIoUScorePlot = []
    metricsCalculator = Metrics(model.numClasses, device)

    for epoch in range(epochs):
        trainingMetrics = trainOneEpoch(model, trainingDataloader, optimizer, criterion, device, metricsCalculator)
        validationMetrics = validateOneEpoch(model, validationDataloader, criterion, device, metricsCalculator)
        scheduler.step(validationMetrics['Loss'])

        # Logging of metrics.
        trainingLossPlot.append(trainingMetrics['Loss'])
        validationLossPlot.append(validationMetrics['Loss'])
        validationDiceScorePlot.append(validationMetrics['Dice Coefficient'])
        validationIoUScorePlot.append(validationMetrics['IoU'])
        logResults(epoch, trainingMetrics, validationMetrics)
    
    # Plot training metrics after training ends, to decrease computational overhead.
    validationDiceScorePlot = [value.cpu().item() for value in validationDiceScorePlot]
    validationIoUScorePlot = [value.cpu().item() for value in validationIoUScorePlot]
    PNGPath = plotMetrics(trainingLossPlot, validationLossPlot, validationDiceScorePlot, validationIoUScorePlot, trialNumber)

    trainingMetrics = {key: value.cpu().item() if isinstance(value, Tensor) else value for key, value in trainingMetrics.items()}
    validationMetrics = {key: value.cpu().item() if isinstance(value, Tensor) else value for key, value in validationMetrics.items()}
    return trainingMetrics, validationMetrics, PNGPath
import torch
import tqdm
from trainingVisualization import logResults, plotMetrics
from Metrics import Metrics

def trainOneEpoch(model, trainingDataloader, optimizer, criterion, device):
    model.train()
    runningLoss = 0

    for imagePair in tqdm.tqdm(trainingDataloader, desc = "Training"):
        # Data shall also be sent to GPU.
        image = imagePair[0].float().to(device)
        groundTruth = imagePair[1].long().to(device)
        yPredicted = model(image)
        optimizer.zero_grad()
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (C, H, W)
        loss = criterion(yPredicted, groundTruth)
        runningLoss += loss.item()
        loss.backward()
        optimizer.step()
    return runningLoss / len(trainingDataloader)

def validateOneEpoch(model, validationDataloader, criterion, device):
    model.eval()
    runningLoss = 0
    diceScore = 0
    IoUScore = 0
    numClasses = model.numClasses

    with torch.no_grad():
        for imagePair in tqdm.tqdm(validationDataloader, desc = "Validation"):
            # Data shall also be sent to GPU.
            image = imagePair[0].float().to(device)
            groundTruth = imagePair[1].long().to(device)
            yPredicted = model(image)
            # Input tensor form: (B, C, H, W)
            # Ground truth tensor form: (C, H, W)
            loss = criterion(yPredicted, groundTruth)
            runningLoss += loss.item()
            # Metrics are calculated during validation.
            diceScore += Metrics.diceCoefficient(yPredicted, groundTruth, numClasses)
            IoUScore += Metrics.IoU(yPredicted, groundTruth, numClasses)
    return runningLoss / len(validationDataloader), diceScore / len(validationDataloader), IoUScore / len(validationDataloader)

def trainingLoop(model, trainingDataloader, validationDataloader, optimizer, criterion, epochs, patience, device):
    bestValidationLoss = float('inf')
    epochsWithoutImprovement = 0
    trainingLossPlot = []
    validationLossPlot = []
    diceScorePlot = []
    IoUScorePlot = []

    for epoch in range(epochs):
        trainingLoss = trainOneEpoch(model, trainingDataloader, optimizer, criterion, device)
        validationLoss, diceScore, IoUScore = validateOneEpoch(model, validationDataloader, criterion, device)

        # Live plotting of metrics.
        trainingLossPlot.append(trainingLoss)
        validationLossPlot.append(validationLoss)
        diceScorePlot.append(diceScore)
        IoUScorePlot.append(IoUScore)
        plotMetrics(trainingLossPlot, validationLossPlot, diceScorePlot, IoUScorePlot, epoch)
        logResults(epoch, trainingLoss, validationLoss, diceScore, IoUScore)

        # Patience implementation.
        if validationLoss < bestValidationLoss:
            bestValidationLoss = validationLoss
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1
            if epochsWithoutImprovement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}.\n")
                break

    return validationLoss, diceScore, IoUScore
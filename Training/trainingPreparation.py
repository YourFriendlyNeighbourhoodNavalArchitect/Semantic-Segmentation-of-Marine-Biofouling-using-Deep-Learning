from tqdm import tqdm
from torch import no_grad
from trainingVisualization import logResults, plotMetrics
from Metrics import Metrics

def trainOneEpoch(model, trainingDataloader, optimizer, criterion, device):
    model.train()
    runningLoss = 0

    for imagePair in tqdm(trainingDataloader, desc = "Training"):
        # Data shall also be sent to GPU.
        image = imagePair[0].to(device)
        groundTruth = imagePair[1].to(device)
        yPredicted = model(image)
        optimizer.zero_grad()
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (B, H, W)
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
    accuracy = 0
    precision = 0
    recall = 0
    numClasses = model.numClasses

    with no_grad():
        for imagePair in tqdm(validationDataloader, desc = "Validation"):
            # Data shall also be sent to GPU.
            image = imagePair[0].to(device)
            groundTruth = imagePair[1].to(device)
            yPredicted = model(image)
            # Input tensor form: (B, C, H, W)
            # Ground truth tensor form: (C, H, W)
            loss = criterion(yPredicted, groundTruth)
            runningLoss += loss.item()
            # Metrics are calculated during validation.
            diceScore += Metrics.diceCoefficient(yPredicted, groundTruth, numClasses)
            IoUScore += Metrics.IoU(yPredicted, groundTruth, numClasses)
            accuracy += Metrics.accuracy(yPredicted, groundTruth)
            precision += Metrics.precision(yPredicted, groundTruth, numClasses)
            recall += Metrics.recall(yPredicted, groundTruth, numClasses)

    return runningLoss / len(validationDataloader), diceScore / len(validationDataloader), IoUScore / len(validationDataloader), accuracy / len(validationDataloader), precision / len(validationDataloader), recall / len(validationDataloader)

def trainingLoop(model, trainingDataloader, validationDataloader, optimizer, scheduler, criterion, epochs, patience, device, trialNumber):
    bestValidationLoss = float('inf')
    epochsWithoutImprovement = 0
    trainingLossPlot = []
    validationLossPlot = []
    diceScorePlot = []
    IoUScorePlot = []

    for epoch in range(epochs):
        trainingLoss = trainOneEpoch(model, trainingDataloader, optimizer, criterion, device)
        validationLoss, diceScore, IoUScore, accuracy, precision, recall = validateOneEpoch(model, validationDataloader, criterion, device)
        scheduler.step()

        # Logging of metrics.
        trainingLossPlot.append(trainingLoss)
        validationLossPlot.append(validationLoss)
        diceScorePlot.append(diceScore)
        IoUScorePlot.append(IoUScore)
        logResults(epoch, trainingLoss, validationLoss, diceScore, IoUScore, accuracy, precision, recall)

        # Patience implementation.
        if validationLoss < bestValidationLoss:
            bestValidationLoss = validationLoss
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1
            if epochsWithoutImprovement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}.\n")
                break
    
    # Plot training metrics after training ends, to decrease computational overhead.
    PNGPath = plotMetrics(trainingLossPlot, validationLossPlot, diceScorePlot, IoUScorePlot, trialNumber)

    return trainingLoss, validationLoss, diceScore, IoUScore, accuracy, precision, recall, PNGPath
from torch import argmax, logical_and
from configurationFile import NUM_CLASSES

def computeMetrics(prediction, groundTruth):
    # Metrics are calculated by evaluating the entire batch at once.
    prediction = argmax(prediction, dim = 1)
    diceScore = 0.0
    IoU = 0.0
    precision = 0.0
    recall = 0.0
    accuracy = (prediction == groundTruth).float().mean()

    # Compute per-class metrics for the entire batch.
    for index in range(NUM_CLASSES):
        predictionMask = (prediction == index)
        groundTruthMask = (groundTruth == index)
        TP = logical_and(predictionMask, groundTruthMask).sum().float()
        FP = logical_and(predictionMask, ~groundTruthMask).sum().float()
        FN = logical_and(~predictionMask, groundTruthMask).sum().float()

        diceScore += (2 * TP) / (2 * TP + FP + FN + 1e-6)
        IoU += TP / (TP + FP + FN + 1e-6)
        precision += TP / (TP + FP + 1e-6)
        recall += TP / (TP + FN + 1e-6)

    # Average class-wise metrics.
    diceScore /= NUM_CLASSES
    IoU /= NUM_CLASSES
    precision /= NUM_CLASSES
    recall /= NUM_CLASSES

    metrics = {'Dice Coefficient': diceScore, 'IoU': IoU, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}
    return metrics
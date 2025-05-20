from torch import argmax
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from configurationFile import NUM_CLASSES

def computeMetrics(prediction, groundTruth):
    # Convert logits to predictions.
    prediction = argmax(prediction, dim = 1)
    
    # Initialize and compute metrics.
    dice = GeneralizedDiceScore(num_classes = NUM_CLASSES, weight_type = 'linear', input_format = 'index')
    IoU = MeanIoU(num_classes = NUM_CLASSES, input_format = 'index')
    accuracy = MulticlassAccuracy(num_classes = NUM_CLASSES, average = 'micro')
    precision = MulticlassPrecision(num_classes = NUM_CLASSES, average = 'macro')
    recall = MulticlassRecall(num_classes = NUM_CLASSES, average = 'macro')
    
    diceScore = dice(prediction, groundTruth)
    IoUScore = IoU(prediction, groundTruth)
    accuracyScore = accuracy(prediction, groundTruth)
    precisionScore = precision(prediction, groundTruth)
    recallScore = recall(prediction, groundTruth)
    
    return {'Dice Coefficient': diceScore.item(), 'IoU': IoUScore.item(), 'Accuracy': accuracyScore.item(),
            'Precision': precisionScore.item(), 'Recall': recallScore.item()}
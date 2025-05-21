from torch import argmax
from torchmetrics.functional.segmentation import generalized_dice_score, mean_iou
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_precision, multiclass_recall
from configurationFile import NUM_CLASSES

def computeMetrics(prediction, groundTruth):
    # Convert logits to predictions.
    prediction = argmax(prediction, dim = 1)
    
    diceScore = generalized_dice_score(prediction, groundTruth, num_classes = NUM_CLASSES, weight_type = 'linear', input_format = 'index')
    IoUScore = mean_iou(prediction, groundTruth, num_classes = NUM_CLASSES, input_format = 'index')
    accuracyScore = multiclass_accuracy(prediction, groundTruth, num_classes = NUM_CLASSES, average = 'macro')
    precisionScore = multiclass_precision(prediction, groundTruth, num_classes = NUM_CLASSES, average = 'macro')
    recallScore = multiclass_recall(prediction, groundTruth, num_classes = NUM_CLASSES, average = 'macro')
    
    return {'Dice Coefficient': diceScore.mean().item(), 'IoU': IoUScore.mean().item(), 'Accuracy': accuracyScore.mean().item(),
            'Precision': precisionScore.mean().item(), 'Recall': recallScore.mean().item()}

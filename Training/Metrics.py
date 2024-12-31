from torch import argmax
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

class Metrics:
    def __init__(self, numClasses, device):
        # Instantiate the metric objects.
        self.diceScore = MulticlassF1Score(numClasses, average = 'macro').to(device)
        self.IoU = MulticlassJaccardIndex(numClasses, average = 'macro').to(device)
        self.accuracy = MulticlassAccuracy(numClasses, average = 'micro').to(device)
        self.precision = MulticlassPrecision(numClasses, average = 'macro').to(device)
        self.recall = MulticlassRecall(numClasses, average = 'macro').to(device)

    def computeMetrics(self, prediction, groundTruth):
        prediction = argmax(prediction, dim = 1)
        metrics = {
            'Dice Coefficient': self.diceScore(prediction, groundTruth),
            'IoU': self.IoU(prediction, groundTruth),
            'Accuracy': self.accuracy(prediction, groundTruth),
            'Precision': self.precision(prediction, groundTruth),
            'Recall': self.recall(prediction, groundTruth)}

        return metrics
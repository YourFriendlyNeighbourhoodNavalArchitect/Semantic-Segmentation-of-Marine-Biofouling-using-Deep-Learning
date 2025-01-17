from torch import argmax
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

class Metrics:
    def __init__(self, numClasses, device):
        # Instantiate the metric objects.
        # Dice coefficient gauges the similarity of two samples [https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient].
        self.diceScore = MulticlassF1Score(numClasses, average = 'macro').to(device)
        # Jaccard index gauges the similarity and diversity of two samples [https://en.wikipedia.org/wiki/Jaccard_index].
        self.IoU = MulticlassJaccardIndex(numClasses, average = 'macro').to(device)
        # Accuracy represents the proportion of correct over total predictions [https://en.wikipedia.org/wiki/Accuracy_and_precision].
        self.accuracy = MulticlassAccuracy(numClasses, average = 'micro').to(device)
        # Precision represents the proportion of true positive over all positive predictions [https://en.wikipedia.org/wiki/Precision_and_recall].
        self.precision = MulticlassPrecision(numClasses, average = 'macro').to(device)
        # Recall represents the proportion of true positive predictions over all positive occurences [https://en.wikipedia.org/wiki/Precision_and_recall].
        self.recall = MulticlassRecall(numClasses, average = 'macro').to(device)

    def computeMetrics(self, prediction, groundTruth):
        prediction = argmax(prediction, dim = 1)
        metrics = {'Dice Coefficient': self.diceScore(prediction, groundTruth), 'IoU': self.IoU(prediction, groundTruth),
                   'Accuracy': self.accuracy(prediction, groundTruth), 'Precision': self.precision(prediction, groundTruth),
                   'Recall': self.recall(prediction, groundTruth)}

        return metrics
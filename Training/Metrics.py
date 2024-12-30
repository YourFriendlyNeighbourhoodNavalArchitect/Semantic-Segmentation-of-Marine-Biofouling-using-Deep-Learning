from torch import argmax

class Metrics:
    # Access to class-specific data is not required.
    @staticmethod
    def diceCoefficient(yPredicted, groundTruth, numClasses):
    # Dice coefficient gauges the similarity of two samples [https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient].
        diceScores = []
        # Selection of the most likely class per pixel.
        yPredicted = argmax(yPredicted, dim = 1)

        for index in range(numClasses):
            yPredictedClass = (yPredicted == index).float()
            groundTruthClass = (groundTruth == index).float()
            
            intersection = (yPredictedClass * groundTruthClass).sum()
            # Ensure computational stability.
            diceScore = (2. * intersection) / (yPredictedClass.sum() + groundTruthClass.sum() + 1e-6)
            diceScores.append(diceScore.item())
        
        # Average of per class metrics.
        return sum(diceScores) / len(diceScores)

    @staticmethod
    def IoU(yPredicted, groundTruth, numClasses):
        # Jaccard index gauges the similarity and diversity of two samples [https://en.wikipedia.org/wiki/Jaccard_index].
        IoUScores = []
        yPredicted = argmax(yPredicted, dim = 1)

        for index in range(numClasses):
            yPredictedClass = (yPredicted == index).float()
            groundTruthClass = (groundTruth == index).float()
            
            intersection = (yPredictedClass * groundTruthClass).sum()
            union = yPredictedClass.sum() + groundTruthClass.sum() - intersection
            IoUScore = intersection / (union + 1e-6)
            IoUScores.append(IoUScore.item())
        
        return sum(IoUScores) / len(IoUScores)

    @staticmethod
    def accuracy(yPredicted, groundTruth):
        # Accuracy represents the proportion of correct over total predictions [https://en.wikipedia.org/wiki/Accuracy_and_precision].
        # Highly class-sensitive.
        yPredicted = argmax(yPredicted, dim = 1)

        correctPixels = (yPredicted == groundTruth).sum().item()
        totalPixels = groundTruth.numel()

        return correctPixels / totalPixels

    @staticmethod
    def precision(yPredicted, groundTruth, numClasses):
        # Precision represents the proportion of true positive over all positive predictions [https://en.wikipedia.org/wiki/Precision_and_recall].
        precisions = []
        yPredicted = argmax(yPredicted, dim = 1)

        for index in range(numClasses):
            yPredictedClass = (yPredicted == index).float()
            groundTruthClass = (groundTruth == index).float()

            truePositives = (yPredictedClass * groundTruthClass).sum()
            predictedPositives = yPredictedClass.sum()

            if predictedPositives == 0:
                precisions.append(0.0)
            else:
                precisions.append((truePositives / predictedPositives).item())

        return sum(precisions) / len(precisions)

    @staticmethod
    def recall(yPredicted, groundTruth, numClasses):
        # Recall represents the proportion of true positive predictions over all positive occurences [https://en.wikipedia.org/wiki/Precision_and_recall].
        recalls = []
        yPredicted = argmax(yPredicted, dim = 1)

        for index in range(numClasses):
            yPredictedClass = (yPredicted == index).float()
            groundTruthClass = (groundTruth == index).float()

            truePositives = (yPredictedClass * groundTruthClass).sum()
            actualPositives = groundTruthClass.sum()

            if actualPositives == 0:
                recalls.append(0.0)
            else:
                recalls.append((truePositives / actualPositives).item())
        
        return sum(recalls) / len(recalls)
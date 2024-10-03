import torch

class Metrics:
    # Access to class-specific data is not required.
    @staticmethod
    def diceCoefficient(yPredicted, groundTruth, numClasses):
    # Dice coefficient gauges the similarity of two samples [https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient].
        diceScores = []
        # Selection of the most likely class per pixel.
        yPredicted = torch.argmax(yPredicted, dim = 1)

        for index in range(numClasses):
            yPredictedClass = (yPredicted == index).float()
            groundTruthClass = (groundTruth == index).float()
            
            intersection = (yPredictedClass * groundTruthClass).sum()
            # Avoiding division by zero.
            diceScore = (2. * intersection) / (yPredictedClass.sum() + groundTruthClass.sum() + 1e-6)
            diceScores.append(diceScore.item())
        
        #Average of class-by-class metrics.
        return sum(diceScores) / len(diceScores)

    # Access to class-specific data is not required.
    @staticmethod
    def IoU(yPredicted, groundTruth, numClasses):
        # Jaccard index gauges the similarity and diversity of two samples [https://en.wikipedia.org/wiki/Jaccard_index].
        IoUScores = []
        # Selection of the most likely class per pixel.
        yPredicted = torch.argmax(yPredicted, dim = 1)

        for index in range(numClasses):
            yPredictedClass = (yPredicted == index).float()
            groundTruthClass = (groundTruth == index).float()
            
            intersection = (yPredictedClass * groundTruthClass).sum()
            union = yPredictedClass.sum() + groundTruthClass.sum() - intersection
            # Avoiding division by zero.
            IoUScore = intersection / (union + 1e-6)
            IoUScores.append(IoUScore.item())
        
        #Average of class-by-class metrics.
        return sum(IoUScores) / len(IoUScores)
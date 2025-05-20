from torch import argmax
from torch.nn import Module, CrossEntropyLoss
from torchmetrics.functional.segmentation import generalized_dice_score
from configurationFile import NUM_CLASSES

class LossFunction(Module):
    # Composite loss function suggested in the relevant literature [https://doi.org/10.1080/08927014.2023.2185143].
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.alpha = alpha
        self.crossEntropyLoss = CrossEntropyLoss()

    def diceLoss(self, prediction, groundTruth):
        prediction = argmax(prediction, dim = 1)
        diceScore = generalized_dice_score(prediction, groundTruth, num_classes = NUM_CLASSES, weight_type = 'linear', input_format = 'index')
        return 1 - diceScore.mean()

    def forward(self, prediction, groundTruth):
        crossEntropyLoss = self.crossEntropyLoss(prediction, groundTruth)
        diceLoss = self.diceLoss(prediction, groundTruth)
        return self.alpha * crossEntropyLoss + (1 - self.alpha) * diceLoss
from torch import argmax
from torch.nn import Module, CrossEntropyLoss
from torchmetrics.segmentation import GeneralizedDiceScore
from configurationFile import NUM_CLASSES

class LossFunction(Module):
    # Composite loss function suggested in the relevant literature [https://doi.org/10.1080/08927014.2023.2185143].
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.alpha = alpha
        self.crossEntropyLoss = CrossEntropyLoss()
        self.dice = GeneralizedDiceScore(num_classes = NUM_CLASSES, weight_type = 'linear', input_format = 'index')

    def diceLoss(self, prediction, groundTruth):
        diceScore = self.dice(argmax(prediction, dim = 1), groundTruth)
        return 1 - diceScore

    def forward(self, prediction, groundTruth):
        crossEntropyLoss = self.crossEntropyLoss(prediction, groundTruth)
        diceLoss = self.diceLoss(prediction, groundTruth)
        return self.alpha * crossEntropyLoss + (1 - self.alpha) * diceLoss
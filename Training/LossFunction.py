from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import softmax, one_hot
from configurationFile import NUM_CLASSES

class LossFunction(Module):
    # Composite loss function suggested in the relevant literature [https://doi.org/10.1080/08927014.2023.2185143].
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.alpha = alpha
        self.crossEntropyLoss = CrossEntropyLoss()

    def diceLoss(self, prediction, groundTruth):
        groundTruth = one_hot(groundTruth, NUM_CLASSES).permute(0, 3, 1, 2).float()
        prediction = softmax(prediction, dim = 1)
        intersection = (prediction * groundTruth).sum(dim = (2, 3))
        union = prediction.sum(dim = (2, 3)) + groundTruth.sum(dim = (2, 3))
        diceScores = (2. * intersection + 1e-6) / (union + 1e-6)
        diceScore = diceScores.mean(dim = 1)
        
        return 1 - diceScore.mean(dim = 0)

    def forward(self, prediction, groundTruth):
        crossEntropyLoss = self.crossEntropyLoss(prediction, groundTruth)
        diceLoss = self.diceLoss(prediction, groundTruth)
        return self.alpha * crossEntropyLoss + (1 - self.alpha) * diceLoss
import torch
from torch.nn import CrossEntropyLoss, Module
from torchmetrics.functional.segmentation import generalized_dice_score

from Various.configurationFile import NUM_CLASSES_new


class LossFunction(Module):
    # Composite loss function suggested in the relevant literature [https://doi.org/10.1080/08927014.2023.2185143].
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.crossEntropyLoss = CrossEntropyLoss()

    def diceLoss(self, prediction: torch.Tensor, groundTruth: torch.Tensor) -> torch.Tensor:
        prediction = torch.argmax(prediction, dim=1)
        diceScore = generalized_dice_score(
            prediction,
            groundTruth,
            num_classes=NUM_CLASSES_new,
            weight_type="linear",
            input_format="index",
        )
        return 1 - diceScore.mean()

    def diceLoss_corrected(self, prediction: torch.Tensor, groundTruth: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(prediction, dim=1)
        groundTruthOneHot = torch.nn.functional.one_hot(
            groundTruth, NUM_CLASSES_new,
        ).permute(0, 3, 1, 2).float()
        # Per-class volumes (for generalized Dice weighting).
        weights = 1.0 / (groundTruthOneHot.sum(dim=(2, 3)) ** 2 + 1e-6)
        intersection = (probs * groundTruthOneHot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + groundTruthOneHot.sum(dim=(2, 3))
        dice = (2 * weights * intersection + 1e-6) / (weights * union + 1e-6)
        return 1 - dice.mean()

    def forward(self, prediction: torch.Tensor, groundTruth: torch.Tensor) -> torch.Tensor:
        crossEntropyLoss = self.crossEntropyLoss(prediction, groundTruth)
        diceLoss = self.diceLoss_corrected(prediction, groundTruth)

        # return  (1 - self.alpha) * diceLoss
        return self.alpha * crossEntropyLoss + (1 - self.alpha) * diceLoss

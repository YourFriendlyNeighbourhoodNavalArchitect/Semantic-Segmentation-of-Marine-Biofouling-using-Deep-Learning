import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
)
from torchmetrics.functional.segmentation import generalized_dice_score, mean_iou

from Various.configurationFile import NUM_CLASSES_new


def computeMetrics(
    prediction: torch.Tensor,
    groundTruth: torch.Tensor,
) -> tuple[dict, dict]:
    prediction = torch.argmax(prediction, dim=1)

    diceScore = generalized_dice_score(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        weight_type="linear",
        input_format="index",
    )
    IoUScore = mean_iou(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        input_format="index",
    )
    accuracyScore = multiclass_accuracy(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        average="macro",
    )
    precisionScore = multiclass_precision(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        average="macro",
    )
    recallScore = multiclass_recall(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        average="macro",
    )

    perClassDice = generalized_dice_score(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        weight_type="linear",
        input_format="index",
        per_class=True,
    ).squeeze(0)
    perClassIoU = mean_iou(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        input_format="index",
        per_class=True,
    ).mean(dim=0)  # Average over batch -> shape (NUM_CLASSES_new,)
    perClassAccuracy = multiclass_accuracy(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        average="none",
    )
    perClassPrecision = multiclass_precision(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        average="none",
    )
    perClassRecall = multiclass_recall(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        average="none",
    )

    macro_metrics = {
        "Dice Coefficient": diceScore.mean().item(),
        "IoU": IoUScore.mean().item(),
        "Accuracy": accuracyScore.item(),
        "Precision": precisionScore.item(),
        "Recall": recallScore.item(),
    }
    per_class_metrics = {
        "Per Class Dice": perClassDice.tolist(),
        "Per Class IoU": perClassIoU.tolist(),
        "Per Class Accuracy": perClassAccuracy.tolist(),
        "Per Class Precision": perClassPrecision.tolist(),
        "Per Class Recall": perClassRecall.tolist(),
    }
    total_metrics = {**macro_metrics, **per_class_metrics}
    return total_metrics

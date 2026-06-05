import torch
from torchmetrics.functional.segmentation import generalized_dice_score
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_confusion_matrix,
)
from Various.configurationFile import NUM_CLASSES_new


def _metricsFromConfusionMatrix(prediction, groundTruth):
    """
    Compute per-class and macro IoU and Dice from the confusion matrix.

    Why not use torchmetrics.mean_iou?
    When a class is absent from a batch, mean_iou returns a sentinel
    value of -1 for that class. Averaging across batches mixes these
    sentinels into the result, producing impossible negative IoU scores.

    The confusion matrix approach avoids this by accumulating raw
    TP/FP/FN counts first, then dividing once. If a class is absent
    (TP = FP = FN = 0), we return 0.0 instead of a sentinel.

    IoU  = TP / (TP + FP + FN)           always in [0, 1]
    Dice = 2*TP / (2*TP + FP + FN)       always in [0, 1]
    """
    cm = multiclass_confusion_matrix(
        prediction, groundTruth, num_classes=NUM_CLASSES_new
    )
    perClassIoU = []
    perClassDice = []
    for c in range(NUM_CLASSES_new):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp  # column sum minus diagonal
        fn = cm[c, :].sum().item() - tp  # row sum minus diagonal
        iouDenom = tp + fp + fn
        diceDenom = 2 * tp + fp + fn
        perClassIoU.append(tp / iouDenom if iouDenom > 0 else 0.0)
        perClassDice.append(2 * tp / diceDenom if diceDenom > 0 else 0.0)
    return perClassIoU, perClassDice


def computeMetrics(prediction, groundTruth):
    prediction = torch.argmax(prediction, dim=1)

    # Generalized Dice (volume-weighted) — used for loss monitoring.
    generalizedDice = generalized_dice_score(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        weight_type="linear",
        input_format="index",
    )
    perClassGeneralizedDice = generalized_dice_score(
        prediction,
        groundTruth,
        num_classes=NUM_CLASSES_new,
        weight_type="linear",
        input_format="index",
        per_class=True,
    )

    # Classification metrics.
    accuracyScore = multiclass_accuracy(
        prediction, groundTruth, num_classes=NUM_CLASSES_new, average="macro"
    )
    precisionScore = multiclass_precision(
        prediction, groundTruth, num_classes=NUM_CLASSES_new, average="macro"
    )
    recallScore = multiclass_recall(
        prediction, groundTruth, num_classes=NUM_CLASSES_new, average="macro"
    )
    perClassAccuracy = multiclass_accuracy(
        prediction, groundTruth, num_classes=NUM_CLASSES_new, average="none"
    )
    perClassPrecision = multiclass_precision(
        prediction, groundTruth, num_classes=NUM_CLASSES_new, average="none"
    )
    perClassRecall = multiclass_recall(
        prediction, groundTruth, num_classes=NUM_CLASSES_new, average="none"
    )

    # IoU and Dice from confusion matrix (always non-negative).
    perClassIoU, perClassDice = _metricsFromConfusionMatrix(prediction, groundTruth)
    macroIoU = sum(perClassIoU) / len(perClassIoU)
    macroDice = sum(perClassDice) / len(perClassDice)

    return {
        "Per Class Generalized Dice": perClassGeneralizedDice.tolist(),
        "Dice Coefficient": generalizedDice.mean().item(),
        "Mean Dice": macroDice,
        "IoU": macroIoU,
        "Accuracy": accuracyScore.item(),
        "Precision": precisionScore.item(),
        "Recall": recallScore.item(),
        "Per Class Dice": perClassDice,
        "Per Class IoU": perClassIoU,
        "Per Class Accuracy": perClassAccuracy.tolist(),
        "Per Class Precision": perClassPrecision.tolist(),
        "Per Class Recall": perClassRecall.tolist(),
    }

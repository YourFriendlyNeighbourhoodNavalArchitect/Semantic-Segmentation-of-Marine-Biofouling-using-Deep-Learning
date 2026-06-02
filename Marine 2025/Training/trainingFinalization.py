from json import dump, load
from pathlib import Path

import torch

from u_net_models.UNet import UNet


def _roundMetrics(metrics: dict[dict]) -> dict[bool | list[float] | float]:
    """Round all values in a metrics dict, handling both scalars and per-class lists."""
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        elif isinstance(value, list):
            rounded[key] = [round(v, 4) for v in value]
        else:
            rounded[key] = value
    return rounded


def saveTrialData(
    epoch: int,
    currentLR: float,
    trainingMetrics: dict,
    validationMetrics: dict,
    trialNumber: int,
    savePath: Path,
):
    # Store all trial data in a JSON file to facilitate subsequent manipulations.
    logPath = savePath / "trialLog.json"
    if logPath.exists():
        with Path.open(logPath, "r") as file:
            studyData = load(file)
    else:
        studyData = {}

    logEntry = {
        "learningRate": round(currentLR, 6),
        "trainingMetrics": _roundMetrics(trainingMetrics),
        "validationMetrics": _roundMetrics(validationMetrics),
    }
    trialKey = f"Trial {trialNumber}"
    epochKey = f"Epoch {epoch}"
    if trialKey not in studyData:
        studyData[trialKey] = {}
    studyData[trialKey][epochKey] = logEntry
    with Path.open(logPath, "w") as file:
        dump(studyData, file, indent=4)


def saveONNX(
    model: UNet,
    device: str,
    inputShape: tuple,
    savePath: Path,
    trialNumber: int,
):
    # ONNX offers framework interoperability and shared optimization.
    dummyInput = torch.randn(inputShape).to(device)
    path = savePath / f"modelTrial{trialNumber}.onnx"
    torch.onnx.export(
        model,
        dummyInput,
        path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["Input"],
        output_names=["Output"],
    )
    print(f"Model saved in ONNX format at {path}.")
    return path


def saveCheckpoint(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    savePath: Path,
    trialNumber: int,
) -> Path:
    """Save a PyTorch checkpoint for later evaluation or resumption."""
    path = savePath / f"checkpointTrial{trialNumber}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"PyTorch checkpoint saved at {path}.")
    return path


def saveResults(
    maxEpochs: int,
    learningRate: float,
    trainingMetrics: dict,
    validationMetrics: dict,
    savePath: Path,
    trialNumber: int,
) -> Path:
    # Fetch performance metrics and hyperparameter values in JSON format.
    results = {
        "trialNumber": trialNumber,
        "maxEpochs": maxEpochs,
        "learningRate": round(learningRate, 6),
        "trainingMetrics": _roundMetrics(trainingMetrics),
        "validationMetrics": _roundMetrics(validationMetrics),
    }

    path = savePath / f"resultsTrial{trialNumber}.json"
    with Path.open(path, "w") as f:
        dump(results, f, indent=4)
    print(f"Results saved at {path}.")
    return path


def cleanupAndRename(
    trialNumber: int,
    savePath: Path,
) -> None:
    """Rename the trial files to "best" variants."""
    mappings = [
        (f"modelTrial{trialNumber}.onnx", "bestModel.onnx"),
        (f"resultsTrial{trialNumber}.json", "bestResults.json"),
        (f"trainingPlot{trialNumber}.png", "bestTrainingPlot.png"),
        (f"checkpointTrial{trialNumber}.pt", "bestCheckpoint.pt"),
    ]
    for src, dst in mappings:
        srcPath = savePath / src
        dstPath = savePath / dst
        if srcPath.exists():
            Path.rename(srcPath, dstPath)
            print(f"Renamed {src} -> {dst}")

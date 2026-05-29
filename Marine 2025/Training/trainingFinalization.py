from os import rename, remove
from torch import randn, save as torchSave
from torch.onnx import export
from json import dump, load
from Various.configurationFile import MODEL_PATH


def _roundMetrics(metrics):
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


def saveTrialData(epoch, currentLR, trainingMetrics, validationMetrics, trialNumber):
    # Store all trial data in a JSON file to facilitate subsequent manipulations.
    logPath = MODEL_PATH / 'trialLog.json'
    if logPath.exists():
        with open(logPath, 'r') as file:
            studyData = load(file)
    else:
        studyData = {}
    
    logEntry = {
        'learningRate': round(currentLR, 6),
        'trainingMetrics': _roundMetrics(trainingMetrics),
        'validationMetrics': _roundMetrics(validationMetrics),
    }
    trialKey = f'Trial {trialNumber}'
    epochKey = f'Epoch {epoch}'
    if trialKey not in studyData:
        studyData[trialKey] = {}
    studyData[trialKey][epochKey] = logEntry
    with open(logPath, 'w') as file:
        dump(studyData, file, indent = 4)


def saveONNX(model, device, inputShape, savePath, trialNumber):
    # ONNX offers framework interoperability and shared optimization.
    dummyInput = randn(inputShape).to(device)
    path = savePath / f'modelTrial{trialNumber}.onnx'
    export(model, dummyInput, path, export_params = True, 
           opset_version = 17, do_constant_folding = True,
           input_names = ['Input'], output_names = ['Output'])
    print(f'Model saved in ONNX format at {path}.')
    return path


def saveCheckpoint(model, optimizer, epoch, metrics, savePath, trialNumber):
    """Save a PyTorch checkpoint for later evaluation or resumption."""
    path = savePath / f'checkpointTrial{trialNumber}.pt'
    torchSave({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    print(f'PyTorch checkpoint saved at {path}.')
    return path


def saveResults(maxEpochs, learningRate, trainingMetrics, validationMetrics, savePath, trialNumber):
    # Fetch performance metrics and hyperparameter values in JSON format.
    results = {
        'trialNumber': trialNumber,
        'maxEpochs': maxEpochs,
        'learningRate': round(learningRate, 6),
        'trainingMetrics': _roundMetrics(trainingMetrics),
        'validationMetrics': _roundMetrics(validationMetrics),
    }

    path = savePath / f'resultsTrial{trialNumber}.json'
    with open(path, 'w') as f:
        dump(results, f, indent = 4)
    print(f'Results saved at {path}.')
    return path


def cleanupAndRename(trialNumber, savePath):
    """Rename the trial files to 'best' variants."""
    mappings = [
        (f'modelTrial{trialNumber}.onnx', 'bestModel.onnx'),
        (f'resultsTrial{trialNumber}.json', 'bestResults.json'),
        (f'trainingPlot{trialNumber}.png', 'bestTrainingPlot.png'),
        (f'checkpointTrial{trialNumber}.pt', 'bestCheckpoint.pt'),
    ]
    for src, dst in mappings:
        srcPath = savePath / src
        dstPath = savePath / dst
        if srcPath.exists():
            rename(srcPath, dstPath)
            print(f'Renamed {src} -> {dst}')
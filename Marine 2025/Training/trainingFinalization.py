from os import rename, remove
from torch import randn
from torch.onnx import export
from json import dump, load
from configurationFile import MODEL_PATH

def saveTrialData(epoch, currentLR, trainingMetrics, validationMetrics, trialNumber):
    # Store all trial data in a JSON file to facilitate subsequent manipulations.
    logPath = MODEL_PATH / 'trialLog.json'
    if logPath.exists():
        with open(logPath, 'r') as file:
            studyData = load(file)
    else:
        studyData = {}
    
    logEntry = {'learningRate': round(currentLR, 6), 'trainingMetrics': {key: round(value, 4) for key, value in trainingMetrics.items()},
                'validationMetrics': {key: round(value, 4) for key, value in validationMetrics.items()}}    
    trialKey = f'Trial {trialNumber}'
    epochKey = f'Epoch {epoch}'
    if trialKey not in studyData:
        studyData[trialKey] = {}
    studyData[trialKey][epochKey] = logEntry
    with open(logPath, 'w') as file:
        dump(studyData, file, indent = 4)

def saveONNX(model, device, inputShape, savePath, trialNumber):
    # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
    # Exporting requires dummy input tensor.
    dummyInput = randn(inputShape).to(device)
    path = savePath / f'modelTrial{trialNumber}.onnx'
    # Constant folding improves efficiency.
    export(model, dummyInput, path, export_params = True, 
           opset_version = 12, do_constant_folding = True,
           input_names = ['Input'], output_names = ['Output'])
    print(f'Model for trial {trialNumber} saved in ONNX format at {path}.')
    return path

def saveResults(trial, maxEpochs, trainingMetrics, validationMetrics, savePath):
    # Fetch performance metrics and hyperparameter values in JSON format for future reference.
    results = {'trialNumber': trial.number, 'maxEpochs': maxEpochs, 'trainingMetrics': trainingMetrics,
               'validationMetrics': validationMetrics, 'hyperparameters': trial.params}

    path = savePath / f'resultsTrial{trial.number}.json'
    with open(path, 'w') as f:
        dump(results, f, indent = 4)
    print(f'Results for trial {trial.number} saved at {path}.')
    return path

def deleteResiduals(savedFiles, bestTrialNumber, savePath):
    bestModelFile = savePath / f'modelTrial{bestTrialNumber}.onnx'
    bestResultsFile = savePath / f'resultsTrial{bestTrialNumber}.json'
    bestPlotFile = savePath / f'trainingPlot{bestTrialNumber}.png'

    for ONNXFile, JSONFile, PNGFile in savedFiles:
        if ONNXFile != bestModelFile:
            remove(ONNXFile)
        if JSONFile != bestResultsFile:
            remove(JSONFile)
        if PNGFile != bestPlotFile:
            remove(PNGFile)
    
    rename(bestModelFile, savePath / 'bestModel.onnx')
    rename(bestResultsFile, savePath / 'bestResults.json')
    rename(bestPlotFile, savePath / 'bestTrainingPlot.png')
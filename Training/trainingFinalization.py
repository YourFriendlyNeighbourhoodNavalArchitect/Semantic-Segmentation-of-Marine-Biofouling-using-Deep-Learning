from os.path import join
from os import rename, remove
from torch import randn
from torch.onnx import export
from json import dump

def saveONNX(model, device, inputShape, savePath, trialNumber):
    # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
    # Exporting requires dummy input tensor.
    dummyInput = randn(inputShape).to(device)
    path = join(savePath, f'modelTrial{trialNumber}.onnx')
    # Constant folding improves efficiency.
    export(model, dummyInput, path, export_params = True, 
                      opset_version = 12, do_constant_folding = True,
                      input_names = ['Input'], output_names = ['Output'])
    print(f"Model for trial {trialNumber} saved in ONNX format at {path}.")
    return path

def saveResults(trial, trainingMetrics, validationMetrics, savePath):
    # Fetch performance metrics and hyperparameter values in JSON format for future reference.
    results = {
        'trialNumber': trial.number,
        'trainingMetrics': trainingMetrics,
        'validationMetrics': validationMetrics,
        'hyperparameters': trial.params
    }

    path = join(savePath, f'resultsTrial{trial.number}.json')
    with open(path, 'w') as f:
        dump(results, f, indent = 4)
    print(f"Results for trial {trial.number} saved at {path}.")
    return path

def deleteResiduals(savedFiles, bestTrialNumber, modelSavePath, trainingPlotSavePath, modelFlag):
    bestModelFile = join(modelSavePath, f'modelTrial{bestTrialNumber}.onnx')
    bestResultsFile = join(modelSavePath, f'resultsTrial{bestTrialNumber}.json')
    bestPlotFile = join(trainingPlotSavePath, f'trainingPlot{bestTrialNumber}.png')

    for ONNXFile, JSONFile, PNGFile in savedFiles:
        if ONNXFile != bestModelFile:
            remove(ONNXFile)
        if JSONFile != bestResultsFile:
            remove(JSONFile)
        if PNGFile != bestPlotFile:
            remove(PNGFile)
    
    if modelFlag:
        rename(bestModelFile, join(modelSavePath, 'bestModel.onnx'))
        rename(bestResultsFile, join(modelSavePath, 'bestResults.json'))
        rename(bestPlotFile, join(trainingPlotSavePath, 'bestTrainingPlot.png'))
    else:
        rename(bestModelFile, join(modelSavePath, 'simpleBestModel.onnx'))
        rename(bestResultsFile, join(modelSavePath, 'simpleBestResults.json'))
        rename(bestPlotFile, join(trainingPlotSavePath, 'simpleBestTrainingPlot.png'))
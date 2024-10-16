import os
import torch
from json import dump

def saveONNX(model, device, inputShape, savePath, trialNumber):
    # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
    # Exporting requires dummy input tensor.
    dummyInput = torch.randn(inputShape).to(device)
    path = os.path.join(savePath, f'modelTrial{trialNumber}.onnx')
    # Constant folding improves efficiency.
    torch.onnx.export(model, dummyInput, path, export_params = True, 
                      opset_version = 12, do_constant_folding = True,
                      input_names = ['Input'], output_names = ['Output'])
    print(f"Model for trial {trialNumber} saved in ONNX format at {path}.")
    return path

def saveResults(trial, savePath):
    # Fetch performance metrics and hyperparameter values in .json format for future reference.
    results = {
        'trialNumber': trial.number,
        'validationLoss': trial.values[0],
        'diceCoefficient': trial.values[1],
        'IoUScore': trial.values[2],
        'hyperparameters': trial.params
    }

    path = os.path.join(savePath, f'resultsTrial{results["trialNumber"]}.json')
    with open(path, 'w') as f:
        dump(results, f, indent = 4)
    print(f"Results for trial {results['trialNumber']} saved at {path}.")
    return path

def deleteResiduals(savedFiles, bestTrialNumber, modelSavePath):
    bestModelFile = os.path.join(modelSavePath, f'modelTrial{bestTrialNumber}.onnx')
    bestHyperparametersFile = os.path.join(modelSavePath, f'hyperparametersTrial{bestTrialNumber}.json')

    for ONNXFile, JSONFile in savedFiles:
        if ONNXFile != bestModelFile:
            os.remove(ONNXFile)
        if JSONFile != bestHyperparametersFile:
            os.remove(JSONFile)

    os.rename(bestModelFile, os.path.join(modelSavePath, 'bestModel.onnx'))
    os.rename(bestHyperparametersFile, os.path.join(modelSavePath, 'bestHyperparameters.json'))

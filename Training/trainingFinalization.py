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

def saveBestHyperparameters(trial, savePath):
    # Fetch optimal hyperparameters in .json format for future reference.
    path = os.path.join(savePath, f'hyperparametersTrial{trial.number}.json')
    with open(path, 'w') as f:
        dump(trial.params, f, indent = 4)
    print(f"Hyperparameters for trial {trial.number} saved at {path}.")
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

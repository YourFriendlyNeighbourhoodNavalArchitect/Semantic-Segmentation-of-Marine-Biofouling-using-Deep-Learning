import os
import json
import torch

def saveONNX(model, device, inputShape, savePath):
        # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
        # Exporting requires dummy input tensor.
        dummyInput = torch.randn(inputShape).to(device)
        path = os.path.join(savePath, 'bestModelOverall.onnx')
        # Constant folding improves efficiency.
        torch.onnx.export(model, dummyInput, path, export_params = True, 
                          opset_version = 12, do_constant_folding = True,
                          input_names = ['Input'], output_names = ['Output'])
        print(f"Best model saved in ONNX format at {path}.")

def saveBestHyperparameters(trial, savePath):
        # Fetch optimal hyperparameters in .json format for future reference.
        path = os.path.join(savePath, 'bestHyperparameters.json')
        with open(path, 'w') as f:
            json.dump(trial.params, f, indent = 4)
        print(f"Hyperparameters for best model saved at {path}.")

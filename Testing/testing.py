import numpy as np
import matplotlib.pyplot as plt
import os
from onnxruntime import InferenceSession
from PIL import Image
from torchvision import transforms
from trainingInitialization import setupDevice

def classIndicesToRGB(prediction):
    # Function that transforms array of indices into RGB mask.
    # Colour map for mask labelling, same as in DatasetVisualizer.py.
    # Green for 'No fouling'.
    # Light yellow for 'Light fouling'.
    # Red for 'Heavy fouling'.
    # Blue for 'Background'.
    classColours = {
        0: [0, 255, 0],
        1: [255, 255, 102],
        2: [255, 0, 0],
        3: [0, 0, 255]
    }

    height, width = prediction.shape
    RGBMask = np.zeros((height, width, 3), dtype = np.uint8)
    for classIndex, colour in classColours.items():
        RGBMask[prediction == classIndex] = colour
    return RGBMask

def testModel(modelPath, testPath, device):
    # Create ONNX Runtime session.
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = InferenceSession(modelPath, providers = providers)

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    # Simple error handling.
    try:
        image = Image.open(testPath).convert('RGB')
        image = transform(image).unsqueeze(0).numpy().astype(np.float32)
        output = session.run(None, {session.get_inputs()[0].name: image})[0]
        prediction = np.argmax(output, axis = 1).squeeze()
    except Exception as e:
        print(f"Error: {e}.")
        return

    RGBMask = classIndicesToRGB(prediction)
    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(image.squeeze(), (1, 2, 0)))
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(RGBMask)
    plt.title("Predicted mask")
    plt.axis('off')

    plt.show()

modelPath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained model\bestModel.onnx'
testFolder = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TESTING\Images'

for fileName in os.listdir(testFolder):
    testPath = os.path.join(testFolder, fileName)
    if os.path.isfile(testPath) and fileName.endswith('.jpg'):
        device = setupDevice()
        testModel(modelPath, testPath, device)

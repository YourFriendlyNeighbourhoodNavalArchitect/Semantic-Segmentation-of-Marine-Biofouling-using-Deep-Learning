import sys

sys.path.append(r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\Training")

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from trainingInitialization import setupDevice

# Colour map for mask labelling, same as in ImageDataset.py.
# Green for 'No fouling'.
# Light yellow for 'Light fouling'.
# Red for 'Heavy fouling'.
# Blue for 'Sea'.
classColors = {
    0: [0, 255, 0],
    1: [255, 255, 102],
    2: [255, 0, 0],
    3: [0, 0, 255]
}

def classIndicesToRGB(prediction):
    # Function that transforms array of indices into RGB mask.
    height, width = prediction.shape
    RGBMask = np.zeros((height, width, 3), dtype = np.uint8)
    for classIndex, color in classColors.items():
        RGBMask[prediction == classIndex] = color
    return RGBMask

def testModel(modelPath, imagePath, device):
    # Create ONNX Runtime session.
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(modelPath, providers = providers)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Simple error handling.
    try:
        image = Image.open(imagePath).convert("RGB")
        image = transform(image).unsqueeze(0).numpy()
        image = image.astype(np.float32)
        output = session.run(None, {session.get_inputs()[0].name: image})[0]
        prediction = np.argmax(output, axis = 1).squeeze()
    except Exception as e:
        print(f"Error loading image: {e}.")
        return

    RGBMask = classIndicesToRGB(prediction)

    plt.imshow(RGBMask)
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.show()

modelPath = r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\OUTPUTS\Trained model\bestModelOverall.onnx"
imagePath = r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\1. Slime\1.15.jpg"
device = setupDevice()
testModel(modelPath, imagePath, device)
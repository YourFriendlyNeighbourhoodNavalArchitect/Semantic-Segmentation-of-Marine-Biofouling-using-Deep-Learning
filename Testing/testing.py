import numpy as np
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from torchvision import transforms
from PIL import Image
from trainingInitialization import setupDevice

def classIndicesToRGB(prediction):
    # Function that transforms array of indices into RGB mask.
    # Colour map for mask labelling, same as in DatasetVisualizer.py.
    # Green for 'No fouling'.
    # Light yellow for 'Light fouling'.
    # Red for 'Heavy fouling'.
    # Blue for 'Sea'.
    classColours = {
        1: [0, 255, 0],
        2: [255, 255, 102],
        3: [255, 0, 0],
        4: [0, 0, 255]
    }

    height, width = prediction.shape
    RGBMask = np.zeros((height, width, 3), dtype = np.uint8)
    for classIndex, colour in classColours.items():
        RGBMask[prediction == classIndex] = colour
    return RGBMask

def testModel(modelPath, imagePath, device):
    # Create ONNX Runtime session.
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = InferenceSession(modelPath, providers = providers)

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

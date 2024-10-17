from numpy import zeros, uint8, float32, argmax
from os import listdir, path
from tkinter import filedialog, messagebox, END, Tk, Label, Entry, Button
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from onnxruntime import InferenceSession
from torch.cuda import is_available

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
    RGBMask = zeros((height, width, 3), dtype = uint8)
    for classIndex, colour in classColours.items():
        RGBMask[prediction == classIndex] = colour
    return RGBMask

def predictMasks(modelPath, inputFolder, outputFolder, device):
    # Predict masks and save them to the output folder.
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = InferenceSession(modelPath, providers = providers)

    transform = Compose([Resize((256, 256)), ToTensor()])

    for fileName in listdir(inputFolder):
        filePath = path.join(inputFolder, fileName)
        if path.isfile(filePath) and fileName.endswith(('.jpg', '.png')):
            try:
                image = Image.open(filePath).convert('RGB')
                image = transform(image).unsqueeze(0).numpy().astype(float32)
                output = session.run(None, {session.get_inputs()[0].name: image})[0]
                prediction = argmax(output, axis = 1).squeeze()

                RGBMask = classIndicesToRGB(prediction)

                maskImage = Image.fromarray(RGBMask)
                maskImage.save(path.join(outputFolder, f'{path.splitext(fileName)[0]}_mask.png'))
            except Exception as e:
                messagebox.showerror('Error', f"Error processing {fileName}: {str(e)}")

    messagebox.showinfo('Success', "Predictions complete! Masks saved.")

# GUI for selecting folders
def selectFolder(entryField):
    folderSelected = filedialog.askdirectory()
    entryField.delete(0, END)
    entryField.insert(0, folderSelected)

def selectFile(entryField):
    fileSelected = filedialog.askopenfilename(filetypes = [('ONNX models', "*.onnx")])
    entryField.delete(0, END)
    entryField.insert(0, fileSelected)

def startPrediction():
    inputFolder = inputFolderEntry.get()
    outputFolder = outputFolderEntry.get()
    modelPath = modelPathEntry.get()

    if not inputFolder or not outputFolder or not modelPath:
        messagebox.showerror('Input Error', "Please provide input/output folders and a model path.")
        return

    device = 'cuda:0' if is_available() else 'cpu'
    predictMasks(modelPath, inputFolder, outputFolder, device)

# GUI setup
root = Tk()
root.title('Marine Biofouling Mask Prediction')
root.geometry('400x400')

# Input folder
inputFolderLabel = Label(root, text = 'Input Folder:')
inputFolderLabel.pack(pady = 5)
inputFolderEntry = Entry(root, width = 40)
inputFolderEntry.pack(pady = 5)
inputFolderButton = Button(root, text = 'Browse', command = lambda: selectFolder(inputFolderEntry))
inputFolderButton.pack(pady = 5)

# Output folder
outputFolderLabel = Label(root, text = 'Output Folder:')
outputFolderLabel.pack(pady = 5)
outputFolderEntry = Entry(root, width = 40)
outputFolderEntry.pack(pady = 5)
outputFolderButton = Button(root, text = 'Browse', command = lambda: selectFolder(outputFolderEntry))
outputFolderButton.pack(pady = 5)

# Model path
modelPathLabel = Label(root, text = 'Model Path:')
modelPathLabel.pack(pady = 5)
modelPathEntry = Entry(root, width = 40)
modelPathEntry.pack(pady = 5)
modelPathButton = Button(root, text = 'Browse', command = lambda: selectFile(modelPathEntry))
modelPathButton.pack(pady = 5)

# Start button
startButton = Button(root, text = 'Start Prediction', command = startPrediction)
startButton.pack(pady = 20)

root.mainloop()
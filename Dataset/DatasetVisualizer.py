import matplotlib.pyplot as plt
import numpy as np
from ImageDataset import ImageDataset

class DatasetVisualizer:
    def __init__(self, numClasses, rootPath, augmentationFlag):
        # Inputs to ImageDataset.py.
        self.numClasses = numClasses
        self.rootPath = rootPath
        self.augmentationFlag = augmentationFlag

        self.dataset = self.loadDataset()
        self.currentIndex = 0
        self.figure, self.axes = plt.subplots(1, 2, figsize = (10, 5))
        self.updatePlot()

        # Move right or left using keyboard arrows, as per method below.
        self.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        plt.show()

    def loadDataset(self):
        # Instantiation of ImageDataset.
        dataset = ImageDataset(self.numClasses, self.rootPath, self.augmentationFlag)
        if len(dataset) == 0:
            raise ValueError("No data pairs located.")
        else:
            print(f"Total number of samples in the dataset: {len(dataset)}")
            return dataset

    def classIndicesToRGB(self, mask):
        # Create a colorized version of the mask.
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

        # Create an empty RGB image for the colorized mask.
        mask = mask.squeeze(0).numpy()
        height, width = mask.shape
        RGBMask = np.zeros((height, width, 3), dtype = np.uint8)
        for classIndex, colour in classColours.items():
            RGBMask[mask == classIndex] = colour
        return RGBMask

    def onKeyPress(self, event):
        if event.key == 'right':
            self.currentIndex = (self.currentIndex + 1) % len(self.dataset)
            self.updatePlot()
        elif event.key == 'left':
            self.currentIndex = (self.currentIndex - 1) % len(self.dataset)
            self.updatePlot()
        else:
            pass

    def updatePlot(self):
        # Utilizing Pyplot, print image and mask side by side.
        image, mask = self.dataset[self.currentIndex]
        title = f"Sample {self.currentIndex + 1}"
        
        self.axes[0].imshow(image.permute(1, 2, 0).numpy())
        self.axes[0].set_title("Image")
        self.axes[0].axis('off')

        RGBMask = self.classIndicesToRGB(mask)
        self.axes[1].imshow(RGBMask)
        self.axes[1].set_title("Mask")
        self.axes[1].axis('off')

        self.figure.canvas.manager.set_window_title(title)
        plt.draw()

numClasses = 4
rootPath = r'C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS\TRAINING'
augmentationFlag = True
DatasetVisualizer(numClasses, rootPath, augmentationFlag)

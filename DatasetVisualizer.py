import matplotlib.pyplot as plt
from ImageDataset import ImageDataset

class DatasetVisualizer:
    def __init__(self, rootPath, augmentationFlag, visualizationFlag, numSamples):
        # Inputs to ImageDataset.py.
        self.rootPath = rootPath
        self.augmentationFlag = augmentationFlag
        self.visualizationFlag = visualizationFlag
        self.numSamples = numSamples
        self.dataset = self.loadDataset()
        self.currentIndex = 0
        self.fig, self.axs = plt.subplots(1, 2, figsize = (10, 5))
        self.updatePlot()
        # Move right or left using keyboard arrows, as per method below.
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        plt.show()

    def loadDataset(self):
        # Instantiation of ImageDataset.
        dataset = ImageDataset(self.rootPath, self.augmentationFlag, self.visualizationFlag)
        print(f"Total number of samples in the dataset: {len(dataset)}")
        return dataset

    def updatePlot(self):
        # Utilizing Pyplot, print image and mask side by side.
        image, mask = self.dataset[self.currentIndex]
        title = f"Sample {self.currentIndex + 1}"
        
        self.axs[0].imshow(image.permute(1, 2, 0).numpy())
        self.axs[0].set_title("Image")
        self.axs[0].axis("off")

        self.axs[1].imshow(mask.permute(1, 2, 0).numpy())
        self.axs[1].set_title("Mask")
        self.axs[1].axis("off")

        self.fig.canvas.manager.set_window_title(title)
        plt.draw()

    def onKeyPress(self, event):
        if event.key == 'right':
            self.currentIndex = (self.currentIndex + 1) % len(self.dataset)
            self.updatePlot()
        elif event.key == 'left':
            self.currentIndex = (self.currentIndex - 1) % len(self.dataset)
            self.updatePlot()

    
rootPath = r"C:\Users\giann\Desktop\NTUA\THESIS\Thesis\INPUTS"
augmentationFlag = True
visualizationFlag = True
numSamples = 5
DatasetVisualizer(rootPath, augmentationFlag, visualizationFlag, numSamples)
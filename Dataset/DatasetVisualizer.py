import numpy as np
from MyDataset import MyDataset
from matplotlib.pyplot import subplots, draw, show
from matplotlib.lines import Line2D
from configurationFile import CLASS_DICTIONARY, TRAINING_PATH, VALIDATION_PATH

class DatasetVisualizer:
    def __init__(self, rootPath):
        self.rootPath = rootPath
        self.dataset = self.loadDataset()
        self.currentIndex = 0
        self.figure, self.axes = subplots(1, 2)
        self.figure.subplots_adjust(left = 0.025, right = 0.975, top = 0.95, bottom = 0.025, wspace = 0.025)
        self.classColors = CLASS_DICTIONARY
        self.updatePlot()
        self.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        show()

    def loadDataset(self):
        dataset = MyDataset(self.rootPath, augmentationFlag = True)
        print(f'Total number of samples in the dataset: {len(dataset)}')
        return dataset

    def calculateClassCoverage(self, mask):
        # Calculate the percentage coverage of each class in the mask.
        try:
            height, width = mask.shape
            totalPixels = height * width
            unique, counts = np.unique(mask, return_counts = True)
            classCoverage = {}

            for className, properties in self.classColors.items():
                classIndex = properties['index']
                classPixels = counts[np.where(unique == classIndex)][0] if classIndex in unique else 0
                classCoverage[className] = (classPixels / totalPixels) * 100
            return classCoverage

        except Exception as e:
            print(f'Error calculating class coverage: {e}')
            raise

    def classIndicesToRGB(self, mask):
        # Map class indices to RGB colors.
        try:
            height, width = mask.shape
            RGBMask = np.zeros((height, width, 3), dtype = np.uint8)
            for _, properties in self.classColors.items():
                classIndex = properties['index']
                color = properties['color']
                RGBMask[mask == classIndex] = color
            
            return RGBMask

        except Exception as e:
            print(f'Error converting mask to RGB: {e}')
            raise

    def onKeyPress(self, event):
        if event.key == 'right':
            self.currentIndex = (self.currentIndex + 1) % len(self.dataset)
            self.updatePlot()
        elif event.key == 'left':
            self.currentIndex = (self.currentIndex - 1) % len(self.dataset)
            self.updatePlot()
        else:
            print('Invalid key pressed. Use left or right arrow keys.')
    
    def generateLegend(self, coverage):
        # Labels appear only for the classes which appear in the mask.
        filteredCoverage = {className: classCoverage for className, classCoverage in coverage.items() if classCoverage > 0}
        legendLabels = [f'{className}: {filteredCoverage[className]:.2f}%' for className in filteredCoverage]
        handles = [Line2D([0], [0], marker = 's', color = 'w', 
                   markerfacecolor = np.array(self.classColors[className]['color']) / 255, 
                   markersize = 6) for className in filteredCoverage]
        return legendLabels, handles

    def updatePlot(self):
        # Utilizing Pyplot, print image and mask side by side.
        image, mask = self.dataset[self.currentIndex]
        image = image.permute(1, 2, 0).numpy()
        RGBMask = self.classIndicesToRGB(mask)
        
        self.axes[0].imshow(image, animated = True)
        self.axes[0].set_title('Image')
        self.axes[0].axis('off')
        self.axes[1].imshow(RGBMask, animated = True)
        self.axes[1].set_title('Mask')
        self.axes[1].axis('off')
        
        # Calculate and display class coverage.
        coverage = self.calculateClassCoverage(mask)
        legendLabels, handles = self.generateLegend(coverage)
        self.axes[1].legend(handles = handles, labels = legendLabels, 
                            loc = 'upper right', title = 'Class Coverage', 
                            title_fontsize = 10, fontsize = 8)

        self.figure.canvas.manager.set_window_title(f'Dataset Visualizer')
        draw()

DatasetVisualizer(TRAINING_PATH)
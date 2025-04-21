import numpy as np
from MyDataset import MyDataset
from matplotlib.pyplot import subplots, draw, show
from matplotlib.lines import Line2D
from onnxruntime import InferenceSession
from trainingInitialization import setupDevice
from configurationFile import CLASS_DICTIONARY, TESTING_PATH, MODEL_PATH

class ModelTester:
    def __init__(self, modelPath, rootPath, device):
        self.modelPath = modelPath
        self.rootPath = rootPath
        self.device = device
        self.dataset = self.loadDataset()
        self.currentIndex = 0
        self.figure, self.axes = subplots(1, 3)
        self.figure.subplots_adjust(left = 0.025, right = 0.975, top = 0.95, bottom = 0.025, wspace = 0.025)
        self.classColors = CLASS_DICTIONARY
        self.session = self.createSession()
        
        self.updatePlot()
        self.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        show()

    def createSession(self):
        # Create ONNX Runtime session.
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            return InferenceSession(self.modelPath, providers = providers)
        
        except Exception as e:
            print(f'Error while creating ONNX session: {e}')
            raise

    def loadDataset(self):
        try:
            dataset = MyDataset(self.rootPath, augmentationFlag = False)
            print(f'Total number of samples in the dataset: {len(dataset)}')
            return dataset
        
        except Exception as e:
            print(f'Error while loading test images: {e}')
            raise

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
            print(f'Error converting prediction to RGB: {e}')
            raise

    def onKeyPress(self, event):
        # Navigate through images with right and left arrow keys.
        if event.key == 'right':
            self.currentIndex = (self.currentIndex + 1) % len(self.testData)
            self.updatePlot()
        elif event.key == 'left':
            self.currentIndex = (self.currentIndex - 1) % len(self.testData)
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
        # Run the model on the testing set.
        image, groundTruth = self.dataset[self.currentIndex]
        imageInput = np.expand_dims(image.numpy().astype(np.float32), axis = 0)
        output = self.session.run(None, {self.session.get_inputs()[0].name: imageInput})[0]
        prediction = np.argmax(output, axis = 1).squeeze()
        predictedMask = self.classIndicesToRGB(prediction)
        groundTruthMask = self.classIndicesToRGB(groundTruth)

        self.axes[0].imshow(image.permute(1, 2, 0).numpy(), animated = True)
        self.axes[0].set_title('Image')
        self.axes[0].axis('off')
        self.axes[1].imshow(predictedMask, animated = True)
        self.axes[1].set_title('Predicted Mask')
        self.axes[1].axis('off')
        self.axes[2].imshow(groundTruthMask, animated = True)
        self.axes[2].set_title('Ground Truth')
        self.axes[2].axis('off')

        # Calculate and display class coverage.
        predictedCoverage = self.calculateClassCoverage(prediction)
        groundTruthCoverage = self.calculateClassCoverage(groundTruth)
        predictedLegendLabels, predictedHandles = self.generateLegend(predictedCoverage)
        groundTruthLegendLabels, groundTruthHandles = self.generateLegend(groundTruthCoverage)
        self.axes[1].legend(handles = predictedHandles, labels = predictedLegendLabels, 
                            loc = 'upper right', title = 'Class Coverage', 
                            title_fontsize = 10, fontsize = 8)
        self.axes[2].legend(handles = groundTruthHandles, labels = groundTruthLegendLabels, 
                            loc = 'upper right', title = 'Class Coverage', 
                            title_fontsize = 10, fontsize = 8)

        self.figure.canvas.manager.set_window_title(f'Model Testing')
        draw()

modelPath = MODEL_PATH / 'bestModel.onnx'
device = setupDevice()
ModelTester(modelPath, TESTING_PATH, device)

import numpy as np
from albumentations.augmentations.geometric.resize import Resize
from matplotlib.pyplot import subplots, draw, show
from os.path import join
from os import listdir
from PIL import Image
from matplotlib.lines import Line2D
from onnxruntime import InferenceSession
from trainingInitialization import setupDevice
from configurationFile import CLASS_DICTIONARY, RESOLUTION, TESTING_PATH, MODEL_PATH

class ModelTester:
    def __init__(self, modelPath, rootPath, device):
        self.modelPath = modelPath
        self.rootPath = rootPath
        self.device = device
        self.testData = self.loadTestData()
        self.currentIndex = 0
        self.figure, self.axes = subplots(1, 2, figsize = (10, 5))
        self.figure.subplots_adjust(left = 0.01, right = 0.99, top = 0.95, bottom = 0.01, wspace = 0.025)
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

    def loadTestData(self):
        # Load test images from the test folder.
        try:
            self.imagePaths = sorted([join(self.rootPath, f) for f in listdir(self.rootPath) if f.endswith('.jpg')])
            print(f'Total number of samples in the dataset: {len(self.imagePaths)}')
            return self.imagePaths
        
        except Exception as e:
            print(f'Error while loading test images: {e}')
            raise

    def prepareImage(self, imagePath):
        image = Image.open(imagePath).convert('RGB')
        transform = Resize(RESOLUTION[0], RESOLUTION[1])
        result = transform(image =  np.array(image))
        resizedImage = result['image']
        imageInput = np.array(resizedImage, dtype = np.float32) / 255.0
        return resizedImage, np.expand_dims(imageInput.transpose(2, 0, 1), axis = 0)

    def calculateClassCoverage(self, mask):
        # Calculate the percentage coverage of each class in the mask.
        try:
            totalPixels = mask.size
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
        image, imageInput = self.prepareImage(self.testData[self.currentIndex])
        output = self.session.run(None, {self.session.get_inputs()[0].name: imageInput})[0]
        prediction = np.argmax(output, axis = 1).squeeze()
        predictedMask = self.classIndicesToRGB(prediction)

        self.axes[0].imshow(image)
        self.axes[0].set_title('Image')
        self.axes[0].axis('off')
        self.axes[1].imshow(predictedMask)
        self.axes[1].set_title('Predicted Mask')
        self.axes[1].axis('off')

        # Calculate and display class coverage.
        predictedCoverage = self.calculateClassCoverage(prediction)
        predictedLegendLabels, predictedHandles = self.generateLegend(predictedCoverage)
        self.axes[1].legend(handles = predictedHandles, labels = predictedLegendLabels, 
                            loc = 'upper right', title = 'Class Coverage', 
                            title_fontsize = 10, fontsize = 8)

        self.figure.canvas.manager.set_window_title(f'Model Testing')
        draw()

modelPath = join(MODEL_PATH, 'simpleBestModel.onnx')
device = setupDevice()
ModelTester(modelPath, TESTING_PATH, device)
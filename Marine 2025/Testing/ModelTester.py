import numpy as np
from MyDataset import MyDataset
from computeMetrics import computeMetrics
from torch import from_numpy
from matplotlib.pyplot import subplots, savefig, close
from matplotlib.lines import Line2D
from onnxruntime import InferenceSession
from trainingInitialization import setupDevice
from configurationFile import CLASS_DICTIONARY, TESTING_PATH, MODEL_PATH

class ModelTester:
    def __init__(self, modelPath, rootPath, device):
        self.modelPath = modelPath
        self.rootPath = rootPath
        self.device = device
        self.outputDirectory = MODEL_PATH / 'Predictions'
        self.dataset = self.loadDataset()
        self.classColors = CLASS_DICTIONARY
        self.session = self.createSession()
        self.runInference()

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
    
    def generateLegend(self, coverage):
        # Labels appear only for the classes which appear in the mask.
        filteredCoverage = {className: classCoverage for className, classCoverage in coverage.items() if classCoverage > 0}
        legendLabels = [f'{className}: {filteredCoverage[className]:.2f}%' for className in filteredCoverage]
        handles = [Line2D([0], [0], marker = 's', color = 'w', 
                   markerfacecolor = np.array(self.classColors[className]['color']) / 255, 
                   markersize = 6) for className in filteredCoverage]
        return legendLabels, handles

    def plotResults(self, image, prediction, groundTruth, index):
        # Plot the image, predicted mask, and ground truth mask.
        predictedMask = self.classIndicesToRGB(prediction)
        groundTruthMask = self.classIndicesToRGB(groundTruth)
        figure, axes = subplots(1, 3, figsize = (24, 8))
        figure.subplots_adjust(left = 0.025, right = 0.975, top = 0.95, bottom = 0.025, wspace = 0.025)
        axes[0].imshow(image.permute(1, 2, 0).numpy(), animated = True)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(predictedMask, animated = True)
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        axes[2].imshow(groundTruthMask, animated = True)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

        predictedCoverage = self.calculateClassCoverage(prediction)
        groundTruthCoverage = self.calculateClassCoverage(groundTruth)
        predictedLegendLabels, predictedHandles = self.generateLegend(predictedCoverage)
        groundTruthLegendLabels, groundTruthHandles = self.generateLegend(groundTruthCoverage)
        axes[1].legend(handles = predictedHandles, labels = predictedLegendLabels, loc = 'upper right', title = 'Class Coverage', 
                       title_fontsize = 16, fontsize = 14)
        axes[2].legend(handles = groundTruthHandles, labels = groundTruthLegendLabels, loc = 'upper right', title = 'Class Coverage', 
                       title_fontsize = 16, fontsize = 14)

        savefig(self.outputDirectory / f'Inference {index + 1}.png', dpi = 600, bbox_inches = 'tight')
        close(figure)

    def runInference(self):
        # Run the model on the testing set.
        aggregatedMetrics = {'Dice Coefficient': 0.0, 'IoU': 0.0, 'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0}
        for i in range(len(self.dataset)):
            image, groundTruth = self.dataset[i]
            imageInput = np.expand_dims(image.numpy().astype(np.float32), axis = 0)
            output = self.session.run(None, {self.session.get_inputs()[0].name: imageInput})[0]
            metrics = computeMetrics(from_numpy(output), groundTruth.unsqueeze(axis = 0))
            aggregatedMetrics = {key: value + metrics[key] for key, value in aggregatedMetrics.items()}
            self.plotResults(image, np.argmax(output, axis = 1).squeeze(), groundTruth, i)

        averagedMetrics = {key: value / len(self.dataset) for key, value in aggregatedMetrics.items()}
        print('\n'.join(f'{key}: {value:.4f}' for key, value in averagedMetrics.items()))

modelPath = MODEL_PATH / 'bestModel.onnx'
device = setupDevice()
ModelTester(modelPath, TESTING_PATH, device)
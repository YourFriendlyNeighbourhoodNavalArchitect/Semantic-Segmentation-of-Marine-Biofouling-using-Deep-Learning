import numpy as np
from json import load
from shutil import copy
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from configurationFile import SEED, NUM_CLASSES, SPLIT_RATIOS, ALL_PATH, METADATA_PATH, TRAINING_PATH, VALIDATION_PATH, TESTING_PATH

class SubsetSplit:
    def __init__(self, metadataPath, rootPath, trainingPath, validationPath, testingPath):
        self.metadataPath = metadataPath
        self.rootPath = rootPath
        self.trainingPath = trainingPath
        self.validationPath = validationPath
        self.testingPath = testingPath
        with open(self.metadataPath, 'r') as file:
            self.metadata = load(file)
            self.totalImages = len(self.metadata)
        self.globalClassDistribution = {i: count / self.totalImages for i, count in self.countClassIndices().items()}
        
        self.splitDataset()

    def countClassIndices(self, IDs = None):
        # Calculate class distribution.
        classIndexCount = {i: 0 for i in range(NUM_CLASSES)}
        relevantMetadata = self.metadata if IDs is None else {str(ID): self.metadata[str(ID)] for ID in IDs}
        for metadata in relevantMetadata.values():
            classIndices = metadata.get('uniqueClassIndices', [])
            for index in classIndices:
                classIndexCount[index] += 1
        return classIndexCount

    def assignToSubsets(self):
        # Assign images to the appropriate subsets, such that:
        # 1) Subsets are stratified with respect to class distributions.
        # 2) Split ratios are approximately maintained.
        IDs = np.array([int(ID) for ID in self.metadata.keys()])
        labels = np.array([[self.countClassIndices([ID])[i] for i in range(NUM_CLASSES)] for ID in IDs])
        stratifierOne = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = SPLIT_RATIOS[1] + SPLIT_RATIOS[2], random_state = SEED)
        trainingIndices, restIndices = next(stratifierOne.split(np.zeros((len(IDs), 1)), labels))
            
        restIDs = IDs[restIndices]
        restLabels = labels[restIndices]
        stratifierTwo = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), 
                                                             random_state = SEED)
        validationIndices, testingIndices = next(stratifierTwo.split(np.zeros((len(restIDs), 1)), restLabels))

        trainingSet = {IDs[i] for i in trainingIndices}
        validationSet = {restIDs[i] for i in validationIndices}
        testingSet = {restIDs[i] for i in testingIndices}
        return trainingSet, validationSet, testingSet

    def validateSplits(self, trainingSet, validationSet, testingSet):
        totalAssigned = len(trainingSet) + len(validationSet) + len(testingSet)
        if totalAssigned != self.totalImages:
            print(f'Mismatch in total assigned images: {totalAssigned} != {self.totalImages}')
            return
        
        # Validate that no overlap has occured.
        if (trainingSet & validationSet) or (trainingSet & testingSet) or (validationSet & testingSet):
            print('Overlap detected between subsets.')
            return

        # Validate split ratios.
        trainingRatio = len(trainingSet) / self.totalImages
        validationRatio = len(validationSet) / self.totalImages
        testingRatio = len(testingSet) / self.totalImages
        print(f'Training Ratio: {trainingRatio}')
        print(f'Validation Ratio: {validationRatio}')
        print(f'Testing Ratio: {testingRatio}')

        # Validate class distribution.
        trainingDistribution = {key: value / len(trainingSet) for key, value in self.countClassIndices(trainingSet).items()}
        validationDistribution = {key: value / len(validationSet) for key, value in self.countClassIndices(validationSet).items()}
        testingDistribution = {key: value / len(testingSet) for key, value in self.countClassIndices(testingSet).items()}
        print(f'Global Class Distribution: {self.globalClassDistribution}')
        print(f'Training Class Distribution: {trainingDistribution}')
        print(f'Validation Class Distribution: {validationDistribution}')
        print(f'Testing Class Distribution: {testingDistribution}')
    
    def copySubset(self, subset, path):
        for ID in subset:
            copy(self.rootPath / 'Images' / f'{ID}.jpg', path / 'Images' / f'{ID}.jpg')
            copy(self.rootPath / 'Masks' / f'{ID}.npy', path / 'Masks' / f'{ID}.npy')

    def splitDataset(self):
        # Split the original dataset into training, validation and testing subsets.
        trainingSet, validationSet, testingSet = self.assignToSubsets()
        self.validateSplits(trainingSet, validationSet, testingSet)
        self.copySubset(trainingSet, self.trainingPath)
        self.copySubset(validationSet, self.validationPath)
        self.copySubset(testingSet, self.testingPath)

SubsetSplit(METADATA_PATH, ALL_PATH, TRAINING_PATH, VALIDATION_PATH, TESTING_PATH)
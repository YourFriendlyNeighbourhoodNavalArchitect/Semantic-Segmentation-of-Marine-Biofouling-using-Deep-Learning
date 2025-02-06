import numpy as np
from json import load
from os.path import join
from shutil import copy
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from configurationFile import SEED, NUM_CLASSES, SPLIT_RATIOS, ALL_PATH, METADATA_PATH, TRAINING_PATH, VALIDATION_PATH

class SubsetSplit:
    def __init__(self, metadataPath, rootPath, trainingPath, validationPath):
        self.metadataPath = metadataPath
        self.rootPath = rootPath
        self.trainingPath = trainingPath
        self.validationPath = validationPath
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
        IDs = [int(ID) for ID in self.metadata.keys()]
        labels = np.array([[self.countClassIndices([ID])[i] for i in range(NUM_CLASSES)] for ID in IDs])

        stratifier = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = SPLIT_RATIOS[1], random_state = SEED)
        trainingIndices, validationIndices = next(stratifier.split(np.zeros((len(IDs), 1)), labels))

        trainingSet = {IDs[i] for i in trainingIndices}
        validationSet = {IDs[i] for i in validationIndices}
        return trainingSet, validationSet

    def validateSplits(self, trainingSet, validationSet):
        totalAssigned = len(trainingSet) + len(validationSet)
        if totalAssigned != self.totalImages:
            print(f'Mismatch in total assigned images: {totalAssigned} != {self.totalImages}')
        
        # Validate that no overlap has occured.
        if trainingSet & validationSet:
            print('Overlap detected between subsets.')

        # Validate split ratios.
        trainingRatio = len(trainingSet) / self.totalImages
        validationRatio = len(validationSet) / self.totalImages
        print(f'Training Ratio: {trainingRatio}')
        print(f'Validation Ratio: {validationRatio}')

        # Validate class distribution.
        trainingDistribution = {key: value / len(trainingSet) for key, value in self.countClassIndices(trainingSet).items()}
        validationDistribution = {key: value / len(validationSet) for key, value in self.countClassIndices(validationSet).items()}
        print(f'Global Class Distribution: {self.globalClassDistribution}')
        print(f'Training Class Distribution: {trainingDistribution}')
        print(f'Validation Class Distribution: {validationDistribution}')            

    def copyToPath(self, filename, path):
        copy(join(self.rootPath, filename), join(path, filename))
    
    def copySubset(self, subset, path):
        for ID in subset:
            imagePath = fr'Images\{ID}.jpg'
            maskPath = fr'Masks\{ID}.npy'
            self.copyToPath(imagePath, path)
            self.copyToPath(maskPath, path)

    def splitDataset(self):
        # Split the original dataset into training and validation subsets.
        trainingSet, validationSet = self.assignToSubsets()
        self.validateSplits(trainingSet, validationSet)
        self.copySubset(trainingSet, self.trainingPath)
        self.copySubset(validationSet, self.validationPath)
        print(f'Training subset: ', list(trainingSet))
        print(f'Validation subset: ', list(validationSet))

SubsetSplit(METADATA_PATH, ALL_PATH, TRAINING_PATH, VALIDATION_PATH)

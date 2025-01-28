from json import load
from os.path import join
from shutil import copy
from math import log
from configurationFile import NUM_CLASSES, SPLIT_RATIOS, ALL_PATH, TRAINING_PATH, VALIDATION_PATH

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

    def parseSimilarity(self, similarityString):
        # Process the "Similarity" property into a set of integer values.
        similaritySet = set()
        for part in similarityString.split(', '):
            if '-' in part:
                start, end = map(int, part.split('-'))
                similaritySet.update(range(start, end + 1))
            else:
                similaritySet.add(int(part))
        return similaritySet

    def countClassIndices(self, IDs = None):
        # Step 1: Calculate class distribution.
        classIndexCount = {i: 0 for i in range(NUM_CLASSES)}
        relevantMetadata = self.metadata if IDs is None else {str(ID): self.metadata[str(ID)] for ID in IDs}
        for metadata in relevantMetadata.values():
            classIndices = metadata.get('UniqueClassIndices', [])
            for index in classIndices:
                classIndexCount[index] += 1
        return classIndexCount

    def groupBySimilarity(self):
        # Step 2: Group images by "Similarity" property.
        similarityGroups = {}
        visitedIDs = set()
        for ID, metadata in self.metadata.items():
            if int(ID) in visitedIDs:
                continue
            currentGroup = {int(ID)}
            similarityString = metadata.get('Similarity')
            if similarityString:
                currentGroup.update(self.parseSimilarity(similarityString))
            groupClassIndexCount = self.countClassIndices(currentGroup)
            groupMetadata = {'classIndices': groupClassIndexCount, 'groupSize': len(currentGroup)}
            similarityGroups[frozenset(currentGroup)] = groupMetadata
            visitedIDs.update(currentGroup)
        
        return similarityGroups
    
    def calculateJSD(self, P, Q):
        # Jensen-Shannon divergence measures the similarity between two distributions.
        # Minimizing the JSD results in maximum similarity [https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence].
        p = [P.get(i, 0) for i in range(NUM_CLASSES)]
        q = [Q.get(i, 0) for i in range(NUM_CLASSES)]
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        
        KL = lambda dist, mid: sum(d * log(d / m) for d, m in zip(dist, mid) if d > 0 and m > 0)
        return (KL(p, m) + KL(q, m)) / 2

    def decisionMaking(self, metadata, trainingSet, validationSet):
        # Helper function to determine the best subset for a given asset.
        # Soft constraints are applied regarding split ratios and JSD.
        bestSubset = None
        bestJSD = float('inf')
        
        for setFlag, currentSet, targetRatio in [('training', trainingSet, SPLIT_RATIOS[0]), ('validation', validationSet, SPLIT_RATIOS[1])]:
            if len(currentSet) / self.totalImages < targetRatio:
                subsetClassCounts = self.countClassIndices(currentSet)
                newSize = len(currentSet) + metadata['groupSize']
                newClassCounts = {i: subsetClassCounts[i] + metadata['classIndices'][i] for i in range(NUM_CLASSES)}
                newDistribution = {i: count / newSize for i, count in newClassCounts.items()}
                
                currentJSD = self.calculateJSD(newDistribution, self.globalClassDistribution)
                if currentJSD < bestJSD:
                    bestJSD = currentJSD
                    bestSubset = setFlag
        
        return bestSubset

    def assignToSubsets(self, similarityGroups):
        trainingSet = set()
        validationSet = set()

        # Prioritize larger groups.
        for IDs, metadata in similarityGroups.items():
            bestSubset = self.decisionMaking(metadata, trainingSet, validationSet)
            if bestSubset == 'training':
                trainingSet.update(IDs)
            else:
                validationSet.update(IDs)

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
        trainingDistribution = {key: value / len(trainingSet) if trainingSet else 0 for key, value in self.countClassIndices(trainingSet).items()}
        validationDistribution = {key: value / len(validationSet) if validationSet else 0 for key, value in self.countClassIndices(validationSet).items()}
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
        # Various requirements must be respected.
        similarityGroups = self.groupBySimilarity()
        trainingSet, validationSet = self.assignToSubsets(similarityGroups)
        self.validateSplits(trainingSet, validationSet)
        self.copySubset(trainingSet, self.trainingPath)
        self.copySubset(validationSet, self.validationPath)
        print(f'Training subset: ', list(trainingSet))
        print(f'Validation subset: ', list(validationSet))

metadataPath = join(ALL_PATH, r'Masks\Metadata.json')
SubsetSplit(metadataPath, ALL_PATH, TRAINING_PATH, VALIDATION_PATH)
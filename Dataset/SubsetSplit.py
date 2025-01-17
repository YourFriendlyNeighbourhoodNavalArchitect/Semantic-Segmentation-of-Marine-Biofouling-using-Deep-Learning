from json import load
from os.path import join
from shutil import copy
from configurationFile import NUM_CLASSES, SPLIT_RATIOS, ALL_PATH, TRAINING_PATH, VALIDATION_PATH, TESTING_PATH

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

    def calculateClassDistribution(self, IDs = None):
        # Step 1: Calculate class distribution.
        classDistribution = {i: 0 for i in range(NUM_CLASSES)}
        relevantMetadata = self.metadata if IDs is None else {str(ID): self.metadata[str(ID)] for ID in IDs}
        totalImages = len(relevantMetadata)
        for metadata in relevantMetadata.values():
            uniqueClasses = metadata.get('UniqueClassIndices', [])
            for classIndex in uniqueClasses:
                classDistribution[classIndex] += 1
        classDistribution = {classIndex : count / totalImages if totalImages > 0 else 0 for classIndex, count in classDistribution.items()}
        return classDistribution

    def countClassIndices(self, IDs):
        # Helper function to count class occurences.
        classIndexCount = {}
        for ID in IDs:
            classIndices = self.metadata[str(ID)].get('UniqueClassIndices', [])
            for index in classIndices:
                classIndexCount[index] = classIndexCount.get(index, 0) + 1
        return classIndexCount

    def groupBySimilarity(self):
        # Step 2: Group images by "Similarity" property.
        similarityGroups = {}
        independentImages = []
        for ID, metadata in self.metadata.items():
            if not metadata.get('Similarity'):
                # Store independent images separately.
                independentImages.append({'ID': int(ID), 'underwater': metadata.get('Underwater'), 
                                          'classIndices': self.countClassIndices([int(ID)])})
                continue

            currentGroup = {int(ID)}
            currentGroup.update(self.parseSimilarity(metadata['Similarity']))
            groupClassIndexCount = self.countClassIndices(currentGroup)
            groupMetadata = {'notUnderwater': any(self.metadata[str(image)].get('Underwater') == 'no' for image in currentGroup),
                             'classIndices': groupClassIndexCount}
            similarityGroups[frozenset(currentGroup)] = groupMetadata

        return similarityGroups, independentImages
    
    def decisionMaking(self, asset, trainingSet, validationSet, testingSet):
        # Helper function to determine the best subset for a given asset.
        # Soft constraints are applied regarding split ratios and class distribution imbalance.
        globalClassDistribution = self.calculateClassDistribution()
        trainingClassDistribution = self.calculateClassDistribution(trainingSet)
        validationClassDistribution = self.calculateClassDistribution(validationSet)
        testingClassDistribution = self.calculateClassDistribution(testingSet)
        bestSubset = None
        bestImbalance = float('inf')
        
        for subset, subsetClassDistribution, targetRatio, currentSet in [('training', trainingClassDistribution, SPLIT_RATIOS[0], trainingSet),
                                                                         ('validation', validationClassDistribution, SPLIT_RATIOS[1], validationSet),
                                                                         ('testing', testingClassDistribution, SPLIT_RATIOS[2], testingSet)]:
            if len(currentSet) / self.totalImages < targetRatio:
                currentImbalance = sum(abs(subsetClassDistribution[i] + (1 if i in asset['classIndices'] else 0) - globalClassDistribution[i]) 
                                       for i in range(NUM_CLASSES))
                if currentImbalance < bestImbalance:
                    bestImbalance = currentImbalance
                    bestSubset = subset

        return bestSubset

    def assignToSubsets(self, similarityGroups, independentImages):
        trainingSet = set()
        validationSet = set()
        testingSet = set()

        # Force-place non-underwater images into testing set.
        notUnderwaterGroups = {group: metadata for group, metadata in similarityGroups.items() if metadata['notUnderwater']}
        for group in notUnderwaterGroups:
            testingSet.update(group)
            del similarityGroups[group]
        # Sort the remaining groups by size.
        sortedGroups = similarityGroups.items()

        for group, metadata in sortedGroups:
            for assetID in group:
                asset = self.metadata[str(assetID)]
                asset['classIndices'] = metadata['classIndices']
                bestSubset = self.decisionMaking(asset, trainingSet, validationSet, testingSet)
                if bestSubset == 'training':
                    trainingSet.add(assetID)
                elif bestSubset == 'validation':
                    validationSet.add(assetID)
                else:
                    testingSet.add(assetID)

        for image in independentImages:
            bestSubset = self.decisionMaking(image, trainingSet, validationSet, testingSet)
            if bestSubset == 'training':
                trainingSet.add(image['ID'])
            elif bestSubset == 'validation':
                validationSet.add(image['ID'])
            else:
                testingSet.add(image['ID'])

        return trainingSet, validationSet, testingSet

    def copyToPath(self, filename, path):
        copy(join(self.rootPath, filename), join(path, filename))

    def splitDataset(self):
        # Split the original dataset into training, validation and testing subsets.
        # Various requirements must be respected.
        similarityGroups, independentImages = self.groupBySimilarity()
        trainingSet, validationSet, testingSet = self.assignToSubsets(similarityGroups, independentImages)
        for ID in trainingSet:
            imagePath = fr'Images\{ID}.jpg'
            maskPath = fr'Masks\{ID}.npy'
            self.copyToPath(imagePath, self.trainingPath)
            self.copyToPath(maskPath, self.trainingPath)
        for ID in validationSet:
            imagePath = fr'Images\{ID}.jpg'
            maskPath = fr'Masks\{ID}.npy'
            self.copyToPath(imagePath, self.validationPath)
            self.copyToPath(maskPath, self.validationPath)
        for ID in testingSet:
            imagePath = fr'Images\{ID}.jpg'
            maskPath = fr'Masks\{ID}.npy'
            self.copyToPath(imagePath, self.testingPath)
            self.copyToPath(maskPath, self.testingPath)

        print(f'Training subset: ', list(trainingSet))
        print(f'Validation subset: ', list(validationSet))
        print(f'Testing subset: ', list(testingSet))

metadataPath = join(ALL_PATH, r'Masks\Metadata.json')
SubsetSplit(metadataPath, ALL_PATH, TRAINING_PATH, VALIDATION_PATH, TESTING_PATH)
import numpy as np
from json import loads, dump
from requests import Session
from os.path import exists, splitext, join
from PIL import Image
from configurationFile import CLASS_DICTIONARY, ALL_PATH, API_KEY

class Labelbox:
    def __init__(self, inputFile, outputDirectory):
        # Labelbox outputs a JSON file with all mask information.
        self.inputFile = inputFile
        self.outputDirectory = outputDirectory
        # Reusable and secure connection for all HTTP requests.
        self.session = Session()
        self.session.headers.update({'Authorization': f'Bearer {API_KEY}'})
        # Dictionary to store metadata for dataset stratification.
        self.metadata = {}
        self.saveMasks()

    def readFile(self):
        if not exists(self.inputFile):
            raise FileNotFoundError(f'File {self.inputFile} not found.')
        with open(self.inputFile, 'r') as f:
            return [loads(line) for line in f]

    def downloadMask(self, URL):
        # Individual class masks are located in respective URLs.
        try:
            response = self.session.get(URL, stream = True)
            response.raise_for_status()
            # Each class mask is a binary file.
            mask = Image.open(response.raw).convert('L')
            return np.array(mask)
        except Exception as e:
            print(f'Error downloading mask from {URL}: {e}')
            return None

    def extractMetadata(self, classifications):
        # Extract "Similarity" property.
        for classification in classifications:
            if classification.get('name') == 'Similarity':
                return classification.get('text_answer', {}).get('content')
        return None

    def processImage(self, entry):
        # Parse JSON file based on its structure.
        dataRow = entry.get('data_row', {})
        image = dataRow.get('external_id')
        ID = splitext(image)[0]
        projects = entry.get('projects', {})
        projectData = list(projects.values())[0]
        labels = projectData.get('labels', [])
        annotations = labels[0].get('annotations', {}).get('objects', [])
        classifications = labels[0].get('annotations', {}).get('classifications', [])

        # Extract metadata.
        similarity = self.extractMetadata(classifications)
        # Initialize mask with appropriate dimensions.
        dummyURL = annotations[0].get('mask', {}).get('url')
        dummy = self.downloadMask(dummyURL)
        if dummy is None:
            print(f'Failed to download mask for {ID}. Skipping entry.')
            return None
        fullMask = np.zeros_like(dummy, dtype = np.uint8)

        uniqueClasses = set()
        for annotation in annotations:
            className = annotation.get('name')
            maskURL = annotation.get('mask', {}).get('url')
            classIndex = CLASS_DICTIONARY.get(className, {}).get('index')
            classMask = self.downloadMask(maskURL)
            if classMask is None:
                print(f'Failed to download mask for {ID}. Skipping entry.')
                return None

            # Populate mask with individual class mask information, using CLASS_DICTIONARY.
            fullMask[classMask > 0] = classIndex
            uniqueClasses.add(classIndex)
        
        self.metadata[ID] = {'uniqueClassIndices': list(uniqueClasses), 'similarity': similarity}
        return ID, fullMask

    def saveMetadata(self):
        # Save the metadata dictionary as a JSON file.
        metadataFilePath = join(self.outputDirectory, 'Metadata.json')
        with open(metadataFilePath, 'w') as f:
            dump(self.metadata, f, indent = 4)
        print(f'Metadata saved to {metadataFilePath}.')

    def saveMasks(self):
        entries = self.readFile()
        for entry in entries:
            result = self.processImage(entry)
            if result is None:
                continue
            ID, fullMask = result

            # Save the mask locally as an NPY file.
            outputPath = join(self.outputDirectory, f'{ID}.npy')
            np.save(outputPath, fullMask)
            print(f'Saved mask for {ID} to {outputPath}.')
        
        self.saveMetadata()

inputFile = join(ALL_PATH, r'Masks\Labels.ndjson')
outputDirectory = join(ALL_PATH, 'Masks')
Labelbox(inputFile, outputDirectory)
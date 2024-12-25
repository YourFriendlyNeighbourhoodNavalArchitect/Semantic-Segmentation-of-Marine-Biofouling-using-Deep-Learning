import numpy as np
from json import loads
from requests import Session
from os.path import exists, splitext, join
from PIL import Image
from configurationFile import CLASS_DICTIONARY, API_KEY, ALL_PATH

class MaskExtraction:
    def __init__(self, inputFile, outputDirectory):
        # Labelbox outputs a .ndjson file with all mask information.
        self.inputFile = inputFile
        self.outputDirectory = outputDirectory
        # Reusable and secure connection for all HTTP requests.
        self.session = Session()
        self.session.headers.update({'Authorization': f'Bearer {API_KEY}'})
        self.saveMasks()

    def readFile(self):
        if not exists(self.inputFile):
            raise FileNotFoundError(f"File {self.inputFile} not found.")
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
            print(f"Error downloading mask from {URL}: {e}")
            return None

    def processImage(self, entry):
        # Parse .ndjson file based on its structure.
        dataRow = entry.get('data_row', {})
        image = dataRow.get('external_id')
        ID = splitext(image)[0]
        projects = entry.get('projects', {})
        projectData = list(projects.values())[0]
        labels = projectData.get('labels', [])
        annotations = labels[0].get('annotations', {}).get('objects', [])

        # Initialize mask with appropriate dimensions.
        dummyURL = annotations[0].get('mask', {}).get('url')
        dummy = self.downloadMask(dummyURL)
        fullMask = np.zeros_like(dummy, dtype = np.uint8)

        for annotation in annotations:
            className = annotation.get('name')
            maskURL = annotation.get('mask', {}).get('url')
            classIndex = CLASS_DICTIONARY.get(className, {}).get('index')

            classMask = self.downloadMask(maskURL)
            if classMask is None:
                print(f"Failed to download mask for class {className}.")
                continue

            # Populate mask with individual class mask information, using CLASS_DICTIONARY.
            fullMask[classMask > 0] = classIndex

        return ID, fullMask

    def saveMasks(self):
        entries = self.readFile()
        for entry in entries:
            result = self.processImage(entry)
            if result is None:
                continue
            ID, fullMask = result

            # Save the mask locally as an .npy file.
            outputPath = join(self.outputDirectory, f'{ID}.npy')
            np.save(outputPath, fullMask)
            print(f"Saved mask for {ID} to {outputPath}.")

inputFile = join(ALL_PATH, r'Masks\Labels.ndjson')
outputDirectory = join(ALL_PATH, 'Masks')
MaskExtraction(inputFile, outputDirectory)
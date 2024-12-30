import numpy as np
from torch import from_numpy
from os import listdir
from os.path import join, basename
from PIL import Image
from torchvision import transforms
from configurationFile import ALL_PATH, RESOLUTION, TRAINING_PATH

class Preprocessing:
    def __init__(self, rootPath, outputPath):
        self.rootPath = rootPath
        self.outputPath = outputPath
        self.imageFolder = join(self.rootPath, 'Images')
        self.maskFolder = join(self.rootPath, 'Masks')

        self.imagePaths = sorted([join(self.imageFolder, f) for f in listdir(self.imageFolder) if f.endswith('.jpg')])
        self.maskPaths = sorted([join(self.maskFolder, f) for f in listdir(self.maskFolder) if f.endswith('.npy')])
        self.dataset = []
        
        # Build dataset by matching image and mask filenames.
        maskLookup = {basename(maskPath).split('.')[0]: maskPath for maskPath in self.maskPaths}
        for imagePath in self.imagePaths:
            image = basename(imagePath).split('.')[0]
            if image in maskLookup:
                self.dataset.append((imagePath, maskLookup[image]))

    def resizeObject(self, object, isMask):
        # Resize to the specified resolution.
        resizeTransform = transforms.Resize(RESOLUTION, interpolation = transforms.InterpolationMode.NEAREST 
                                            if isMask else transforms.InterpolationMode.BILINEAR)
        return resizeTransform(object)

    def applyFlip(self, image, mask):
        # Apply horizontal or vertical flips.
        randomFlag = int(np.random.choice([1, 2, 3]))
        if randomFlag == 1:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        elif randomFlag == 2:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        else:
            image = transforms.functional.hflip(image)
            image = transforms.functional.vflip(image)
            mask = transforms.functional.hflip(mask)
            mask = transforms.functional.vflip(mask)

        return image, mask

    def applyRotation(self, image, mask):
        # Apply a random rotation by 90 or 270 degrees.
        randomAngle = int(np.random.choice([90, 270]))
        image = transforms.functional.rotate(image, randomAngle)
        mask = transforms.functional.rotate(mask, randomAngle)
        return image, mask
    
    def save(self):
        # Essentialy triple the dataset size.
        nameCounter = 1

        for imagePath, maskPath in self.dataset:
            image = Image.open(imagePath).convert('RGB')
            mask = np.load(maskPath)
            mask = from_numpy(mask).long().unsqueeze(0)

            image = self.resizeObject(image, isMask = False)
            mask = self.resizeObject(mask, isMask = True)

            imageSavePath = join(self.outputPath, f'Images/{nameCounter}.jpg')
            maskSavePath = join(self.outputPath, f'Masks/{nameCounter}.npy')
            image.save(imageSavePath)
            np.save(maskSavePath, mask.squeeze(0).numpy())

            flippedImage, flippedMask = self.applyFlip(image, mask)
            flippedImageSavePath = join(self.outputPath, f'Images/{nameCounter + 1}.jpg')
            flippedMaskSavePath = join(self.outputPath, f'Masks/{nameCounter + 1}.npy')
            flippedImage.save(flippedImageSavePath)
            np.save(flippedMaskSavePath, flippedMask.squeeze(0).numpy())

            rotatedImage, rotatedMask = self.applyRotation(image, mask)
            rotatedImageSavePath = join(self.outputPath, f'Images/{nameCounter + 2}.jpg')
            rotatedMaskSavePath = join(self.outputPath, f'Masks/{nameCounter + 2}.npy')
            rotatedImage.save(rotatedImageSavePath)
            np.save(rotatedMaskSavePath, rotatedMask.squeeze(0).numpy())

            nameCounter += 3

preprocessing = Preprocessing(ALL_PATH, TRAINING_PATH)
preprocessing.save()
import numpy as np
from torch import from_numpy
from os.path import join, basename, splitext
from os import listdir, remove
from PIL import Image
from torchvision import transforms
from configurationFile import RESOLUTION, TRAINING_PATH, VALIDATION_PATH

class Preprocessing:
    def __init__(self, rootPath):
        self.rootPath = rootPath
        self.imageFolder = join(self.rootPath, 'Images')
        self.maskFolder = join(self.rootPath, 'Masks')
        self.imagePaths = sorted([join(self.imageFolder, f) for f in listdir(self.imageFolder) if f.endswith('.jpg')])
        self.maskPaths = sorted([join(self.maskFolder, f) for f in listdir(self.maskFolder) if f.endswith('.npy')])
        self.dataset = [(imagePath, maskPath) for imagePath, maskPath in zip(self.imagePaths, self.maskPaths)
                        if basename(imagePath).replace('.jpg', '.npy') == basename(maskPath)]

    def resizeAsset(self, asset, isMask):
        # Resize to the specified resolution
        resizeTransform = transforms.Resize(RESOLUTION, interpolation = transforms.InterpolationMode.NEAREST 
                                            if isMask else transforms.InterpolationMode.BILINEAR)
        return resizeTransform(asset)

    def applyFlip(self, image, mask):
        # Apply horizontal or vertical flips
        flipTransforms = {1: transforms.functional.hflip, 2: transforms.functional.vflip,
                          3: lambda x: transforms.functional.vflip(transforms.functional.hflip(x))}
        randomFlag = int(np.random.choice([1, 2, 3]))
        image = flipTransforms[randomFlag](image)
        mask = flipTransforms[randomFlag](mask)
        return image, mask

    def applyRotation(self, image, mask):
        # Apply a random rotation by 90 or 270 degrees
        randomAngle = int(np.random.choice([90, 270]))
        image = transforms.functional.rotate(image, randomAngle)
        mask = transforms.functional.rotate(mask, randomAngle)
        return image, mask

    def applyColorJitter(self, image):
        # Apply random color jitter (brightness, contrast, saturation and hue).
        colorJitter = transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1)
        return colorJitter(image)

    def processSubsets(self, augmentationFlag):
        # Only augment the training subset.
        # Validation and testing set shall merely be resized.
        augmentations = [(None, None, '.1'), (self.applyFlip, self.applyColorJitter, '.2'), (self.applyRotation, self.applyColorJitter, '.3')]
    
        for imagePath, maskPath in self.dataset:
            image = Image.open(imagePath).convert('RGB')
            mask = np.load(maskPath)
            mask = from_numpy(mask).long().unsqueeze(0)
            ID = splitext(basename(imagePath))[0]
            image = self.resizeAsset(image, isMask = False)
            mask = self.resizeAsset(mask, isMask = True)
            
            if augmentationFlag:
                for augmentation, jitter, suffix in augmentations:
                    if augmentation is None:
                        augmentedImage, augmentedMask = image, mask
                    else:
                        augmentedImage, augmentedMask = augmentation(image, mask)
                        augmentedImage = jitter(augmentedImage)
                
                    imageSavePath = join(self.rootPath, f'Images/{ID}{suffix}.jpg')
                    maskSavePath = join(self.rootPath, f'Masks/{ID}{suffix}.npy')
                    augmentedImage.save(imageSavePath)
                    np.save(maskSavePath, augmentedMask.squeeze(0).numpy())
            
                # Remove original files after successful augmentation.
                remove(imagePath)
                remove(maskPath)
            else:
                image.save(imagePath)
                np.save(maskPath, mask.squeeze(0).numpy())

subset = Preprocessing(VALIDATION_PATH)
subset.processSubsets(augmentationFlag = False)
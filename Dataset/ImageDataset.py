import os
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, rootPath, augmentationFlag, visualizationFlag):
        # Augmentation flag handles data augmentation for training.
        # Visualization flag handles dataset visualization in DatasetVisualizer.py.
        self.rootPath = rootPath
        self.imagePath = os.path.join(self.rootPath, 'Images')
        self.maskPath = os.path.join(self.rootPath, 'Masks')

        self.images = sorted([f for f in os.listdir(self.imagePath) if f.endswith(('.png', '.jpg'))])

        self.augmentTransform = transforms.Compose([
            # Image size cannot be further increased due to hardware bottleneck.
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)])
        
        self.augmentationFlag = augmentationFlag
        self.visualizationFlag = visualizationFlag

    def __len__(self):
        return len(self.images)

    def loadMask(self, maskBaseName, numClasses):
        # Colour map for mask labelling.
        # Green for 'No fouling'.
        # Light yellow for 'Light fouling'.
        # Red for 'Heavy fouling'.
        # Blue for 'Sea'.
        colours = {
            0: [0, 255, 0],
            1: [255, 255, 102],
            2: [255, 0, 0],
            3: [0, 0, 255]
        }

        if self.visualizationFlag:
            # RGB for visualization.
            combinedMask = np.zeros((256, 256, 3), dtype = np.uint8)

            for i in range(numClasses):
                # Utilizing the standardized names created by renameFiles.py.
                classMaskPath = os.path.join(self.maskPath, f'{maskBaseName}.{i + 1}.npy')
            
                if os.path.exists(classMaskPath):
                    classMask = np.load(classMaskPath)
                    classMask = Image.fromarray(classMask.astype(np.uint8)).resize((256, 256), Image.NEAREST)
                    classMask = np.array(classMask)

                    colour = colours[i]
                    combinedMask[classMask != 0] = colour

            if np.all(combinedMask == 0):
                raise FileNotFoundError(f"No masks found for image {maskBaseName}.")

            combinedMask = Image.fromarray(combinedMask)
            return combinedMask

        else:
            # Long tensor for training.
            classMask = torch.zeros((256, 256), dtype = torch.long)

            for i in range(numClasses):
                # Utilizing the standardized names created by renameFiles.py.
                classMaskPath = os.path.join(self.maskPath, f'{maskBaseName}.{i + 1}.npy')
            
                if os.path.exists(classMaskPath):
                    classMaskData = np.load(classMaskPath)
                    classMaskData = Image.fromarray(classMaskData.astype(np.uint8)).resize((256, 256), Image.NEAREST)
                    classMaskData = np.array(classMaskData)

                    classMask[classMaskData != 0] = i

            if torch.all(classMask == 0):
                raise FileNotFoundError(f"No masks found for base name {maskBaseName}.")

            return classMask

    def applyTransforms(self, image, mask):
        # Augmentation shall be applied to the image and its mask in the same manner.
        seed = random.randint(0, 2**32)

        torch.manual_seed(seed)
        image = self.augmentTransform(image)

        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.long:
                mask = mask.byte().numpy()
            mask = Image.fromarray(mask)
    
        torch.manual_seed(seed)
        mask = self.augmentTransform(mask)

        return image, mask

    def __getitem__(self, index):
        # Getter ensures masks are in the correct form, based on the flags.
        imageName = self.images[index]
        maskBaseName = os.path.splitext(imageName)[0]

        image = Image.open(os.path.join(self.imagePath, imageName)).convert('RGB')
        mask = self.loadMask(maskBaseName, 4)

        if self.augmentationFlag:
            image, mask = self.applyTransforms(image, mask)
        else:
            image = transforms.Resize((256, 256))(image)

        image = transforms.ToTensor()(image)

        if self.visualizationFlag:
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.tensor(np.array(mask), dtype = torch.long)

        return image, mask
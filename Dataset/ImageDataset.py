import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, numClasses, rootPath, augmentationFlag):
        # Augmentation flag handles data augmentation for training.
        self.numClasses = numClasses
        self.rootPath = rootPath
        self.imagePath = os.path.join(self.rootPath, 'Images')
        self.maskPath = os.path.join(self.rootPath, 'Masks')
        self.augmentationFlag = augmentationFlag

        # Initialize images list with only those having corresponding masks.
        self.images = []
        for imageFile in sorted(os.listdir(self.imagePath)):
            if imageFile.endswith(('.png', '.jpg')):
                maskBaseName = os.path.splitext(imageFile)[0]
                # Check if any mask files exist for this image.
                if any(os.path.exists(os.path.join(self.maskPath, f'{maskBaseName}.{classIndex + 1}.npy')) for classIndex in range(numClasses)):
                    self.images.append(imageFile)

    def __len__(self):
        return len(self.images)

    def loadMask(self, maskBaseName, numClasses):
        # Long tensor for training.
        classMask = np.zeros((256, 256), dtype = np.int32)

        for classIndex in range(numClasses):
            # Utilizing the standardized names created by renameFiles.py.
            classMaskPath = os.path.join(self.maskPath, f'{maskBaseName}.{classIndex + 1}.npy')
            
            if os.path.exists(classMaskPath):
                classMaskData = np.load(classMaskPath)
                classMaskData = cv2.resize(classMaskData, (256, 256), interpolation = cv2.INTER_NEAREST)
                classMask[classMaskData != 0] = classIndex

        return Image.fromarray(classMask)
    
    def defineTransforms(self, image, angle):
        # Artificially enrich the dataset.
        augmentTransform = transforms.Compose([
                           transforms.Resize((256, 256)),
                           # Image size cannot be further increased due to hardware bottleneck.
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomVerticalFlip(),
                           transforms.Lambda(lambda x: transforms.functional.rotate(x, angle))])
        return augmentTransform(image)

    def applyTransforms(self, image, mask):
        # Augmentation shall be applied to the image and its mask in the same manner.
        seed = np.random.randint(0, 2**31)
        randomAngle = int(np.random.choice([0, 90, 180, 270]))
        torch.manual_seed(seed)
        image = self.defineTransforms(image, randomAngle)
        torch.manual_seed(seed)
        mask = self.defineTransforms(mask, randomAngle)
        return image, mask

    def __getitem__(self, index):
        # Getter ensures masks are in the correct form, based on the flag.
        imageName = self.images[index]
        maskBaseName = os.path.splitext(imageName)[0]

        image = Image.open(os.path.join(self.imagePath, imageName)).convert('RGB')
        mask = self.loadMask(maskBaseName, self.numClasses)

        if self.augmentationFlag:
            image, mask = self.applyTransforms(image, mask)
        else:
            image = transforms.Resize((256, 256))(image)

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

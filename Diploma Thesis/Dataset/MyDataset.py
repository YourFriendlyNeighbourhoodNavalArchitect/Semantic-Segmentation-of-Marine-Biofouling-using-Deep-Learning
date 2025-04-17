import numpy as np
import albumentations as A
from torch import from_numpy, float32
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from configurationFile import RESOLUTION

class MyDataset(Dataset):
    def __init__(self, rootPath, augmentationFlag):
        self.rootPath = rootPath
        self.imageFolder = self.rootPath / 'Images'
        self.maskFolder = self.rootPath / 'Masks'
        self.augmentationFlag = augmentationFlag

        # Implement lazy loading of files to reduce computational overhead.
        self.imagePaths = sorted(list(self.imageFolder.glob('*.jpg')))
        self.maskPaths = sorted(list(self.maskFolder.glob('*.npy')))
        self.dataset = [(imagePath, maskPath) for imagePath, maskPath in zip(self.imagePaths, self.maskPaths)
                        if imagePath.name.replace('.jpg', '.npy') == maskPath.name]
        
        # Define transformation pipeline.
        if self.augmentationFlag:
            self.transformCompose = A.Compose([A.RandomCrop(width = RESOLUTION[0], height = RESOLUTION[1], p = 0.5),
                                               A.Resize(RESOLUTION[0], RESOLUTION[1]),
                                               A.OneOf([A.Rotate(limit = (90, 90), p = 1.0), A.Rotate(limit = (180, 180), p = 1.0),
                                                        A.Rotate(limit = (270, 270), p = 1.0)], p = 0.5),
                                               A.ColorJitter(brightness = 0.05, contrast = 0.05, saturation = 0.05, hue = 0.05, p = 0.5)],
                                               additional_targets = {'mask': 'mask'})
        else:
            self.transformCompose = A.Compose([A.Resize(RESOLUTION[0], RESOLUTION[1])], additional_targets = {'mask': 'mask'})

        self.toTensor = Compose([ToImage(), ToDtype(float32, scale = True)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Getter ensures masks are in the correct form.
        imagePath, maskPath = self.dataset[index]
        image = np.array(Image.open(imagePath).convert('RGB'))
        mask = np.load(maskPath).astype(np.uint8)
        result = self.transformCompose(image = image, mask = mask)
        transformedImage = result['image']
        transformedMask = result['mask']

        imageTensor = self.toTensor(transformedImage)
        maskTensor = from_numpy(transformedMask).long()
        return imageTensor, maskTensor

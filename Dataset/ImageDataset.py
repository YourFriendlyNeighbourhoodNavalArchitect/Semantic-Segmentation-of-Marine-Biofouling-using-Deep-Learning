import numpy as np
from torch import from_numpy
from os import listdir
from os.path import join, basename
from torch import manual_seed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, rootPath):
        self.rootPath = rootPath
        self.imageFolder = join(self.rootPath, 'Images')
        self.maskFolder = join(self.rootPath, 'Masks')

        # Implement lazy loading of files to reduce computational overhead.
        self.imagePaths = sorted([join(self.imageFolder, f) for f in listdir(self.imageFolder) if f.endswith('.jpg')])
        self.maskPaths = sorted([join(self.maskFolder, f) for f in listdir(self.maskFolder) if f.endswith('.npy')])
        # TEMPORARY
        # self.dataset = [(imagePath, maskPath) for imagePath, maskPath in zip(self.imagePaths, self.maskPaths)
        #                 if basename(imagePath).replace('.jpg', '.npy') == basename(maskPath)]
        self.dataset = []
        maskLookup = {basename(maskPath).split('.')[0]: maskPath for maskPath in self.maskPaths}
        for imagePath in self.imagePaths:
            image = basename(imagePath).split('.')[0]
            if image in maskLookup:
                self.dataset.append((imagePath, maskLookup[image]))

        # Define augmentation transformations.
        self.augmentationTransform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()])
    
    def __len__(self):
        return len(self.dataset)

    def resizeObject(self, object, isMask):
        resizeTransform = transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST 
                                            if isMask else transforms.InterpolationMode.BILINEAR)
        return resizeTransform(object)

    def applyAugmentation(self, image, mask):
        # Augmentation shall be applied to the image and its mask in the same manner.
        seed = np.random.randint(0, 2**31)
        randomAngle = int(np.random.choice([0, 90, 180, 270]))

        manual_seed(seed)
        image = self.augmentationTransform(image)
        manual_seed(seed)
        mask = self.augmentationTransform(mask)
        image = transforms.functional.rotate(image, randomAngle)
        mask = transforms.functional.rotate(mask, randomAngle)

        return image, mask

    def __getitem__(self, index):
        # Getter ensures masks are in the correct form.
        imagePath, maskPath = self.dataset[index]
        image = Image.open(imagePath).convert('RGB')
        mask = np.load(maskPath)
        mask = from_numpy(mask).long().unsqueeze(0)

        image = self.resizeObject(image, isMask = False)
        mask = self.resizeObject(mask, isMask = True)
        image, mask = self.applyAugmentation(image, mask)
        image = transforms.ToTensor()(image)
        mask = mask.squeeze(0)

        return image, mask
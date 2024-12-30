import numpy as np
from os import listdir
from os.path import join, basename
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
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Getter ensures masks are in the correct form.
        imagePath, maskPath = self.dataset[index]
        image = Image.open(imagePath).convert('RGB')
        mask = np.load(maskPath)
        image = transforms.ToTensor()(image)

        return image, mask
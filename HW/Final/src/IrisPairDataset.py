import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

from pathlib import Path

from segmentation import segment_and_unwrap

class IrisPairDataset(Dataset):
    def __init__(self, file_path, patch = False, segmentation = False):
        self.pairs = []
        self.segmentation = segmentation
        with open(file_path, 'r') as f:
            for line in f:
                p1, p2, label = line.strip().split(',')
                p1 = Path(p1)
                p2 = Path(p2)
                if patch is True:
                    p1 = str(Path("dataset/patch") / Path(p1.relative_to("dataset")))
                    p2 = str(Path("dataset/patch") / Path(p2.relative_to("dataset")))
                else:
                    p1 = str(Path("dataset/origin") / Path(p1.relative_to("dataset")))
                    p2 = str(Path("dataset/origin") / Path(p2.relative_to("dataset")))
                self.pairs.append((p1.strip(), p2.strip(), float(label)))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            raise FileNotFoundError(f"Cannot read image: {path1}")
        if img2 is None:
            raise FileNotFoundError(f"Cannot read image: {path2}")

        if self.segmentation is True:
            img1 = segment_and_unwrap(img1, (64, 512))
            img2 = segment_and_unwrap(img2, (64, 512))

        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
import torch
from torchvision import transforms
from dataclasses import dataclass
from torch.utils.data import Dataset


def get_transform(data):
    mean = torch.mean(data).item()
    std = torch.std(data).item()
    return transforms.Compose(
        [
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@dataclass
class TransformDataset(Dataset):
    data: Dataset
    transform: transforms.Compose

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform != None:
            transformed_image = self.transform(sample[0])
        return transformed_image, sample[1]

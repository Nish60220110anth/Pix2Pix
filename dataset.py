import torch
from torch.utils.data import Dataset
import os
import os.path as path
from PIL import Image
import config
import numpy as np

class MapDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

    def __getitem__(self, index):
        image_file_path = self.files[index]
        full_path = path.join(self.root_dir, image_file_path)
        image = Image.open(full_path).convert("RGB")
        image = np.array(image)

        input = image[:, :600, :]
        target = image[:, 600:, :]

        aug = config.both_transform(image=input, image0=target)

        input = aug["image"]
        target = aug["image0"]

        input = config.transform_only_input(image=input)["image"]
        target = config.transform_only_mask(image=target)["image"]

        return input, target

    def __len__(self):
        return len(self.files)

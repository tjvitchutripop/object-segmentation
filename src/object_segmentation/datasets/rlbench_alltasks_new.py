import lightning as L
import torch
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as T
from torchvision.datasets import VisionDataset
import os
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pickle
from PIL import Image


class RLBenchAllTasks(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``rlbench-shape-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    base_folder = "rlbench-alltasks-15-py"
    # train_range = (0,873)
    # test_range = (873,969)

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []
        files = os.listdir(os.path.join(self.root, self.base_folder))
        files_sorted = sorted(files)
        self.train_set_indices = []
        self.val_set_indices = []
        self.test_set_indices = []
        # now load the picked numpy arrays
        for idx, file in enumerate(files_sorted):
            entry = np.load(os.path.join(self.root, self.base_folder,file))
            num = int(file.split("-")[1])
            if num < 10:
                self.train_set_indices.append(idx)
            elif num < 13:  
                self.val_set_indices.append(idx)
            else:
                self.test_set_indices.append(idx)
            self.data.append(entry["obs"])
            self.targets.append(entry["pseudo_gt_1"])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"



class RLBenchAllTasksDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size, num_workers):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def prepare_data(self):
    #     # Anything that needs to be done to download.
    #     tv.datasets.CIFAR10(self.root, train=True, download=True)
    #     tv.datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage: str):
        # Set up data augmentation.
        train_transform = T.Compose(
            [
                # T.RandomHorizontalFlip(),
                # T.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(
                    [0.49139968, 0.48215841, 0.44653091],
                    [0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    [0.49139968, 0.48215841, 0.44653091],
                    [0.24703223, 0.24348513, 0.26158784],
                ),
            ]
        )

        # We want to split the training set into train and val. But we don't want transforms on val.
        # So we create two datasets, and make sure that the split is consistent between them.
        dataset = RLBenchAllTasks(
            self.root, train=True, transform=train_transform
        )
        self.train_set = torch.utils.data.Subset(dataset, dataset.train_set_indices)
        self.val_set = torch.utils.data.Subset(dataset, dataset.val_set_indices)
        self.test_set = torch.utils.data.Subset(dataset, dataset.test_set_indices)

        # generator = torch.Generator().manual_seed(42)
        # self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
        #     dataset, [679, 194, 96], generator=generator
        # )


    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return [
            data.DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
            data.DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            ),
        ]

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

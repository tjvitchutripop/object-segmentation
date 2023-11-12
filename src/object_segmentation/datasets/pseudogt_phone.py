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


class PseudoGTPhone(VisionDataset):
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

    base_folder = "pseudogt-phone-py"
    train_range = (0,450)
    test_range = (450,500)

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            data_range = range(self.train_range[0],self.train_range[1])
        else:
            data_range = range(self.test_range[0],self.test_range[1])

        self.data: Any = []
        self.targets = []
        # now load the picked numpy arrays
        for idx in data_range:
            file_path = os.path.join(self.root, self.base_folder, "phone_on_base-method1-"+str(idx)+"-action_object.npz")
            entry = np.load(file_path)
            self.data.append(entry["rgb"])
            self.targets.append(entry["pseudo_gt"])

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    #     self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    #     with open(path, "rb") as infile:
    #         data = pickle.load(infile, encoding="latin1")
    #         self.classes = data[self.meta["key"]]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

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



class PseudoGTPhoneDataModule(L.LightningDataModule):
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
        train_dataset = PseudoGTPhone(
            self.root, train=True, transform=train_transform
        )
        val_dataset = PseudoGTPhone(
            self.root, train=True, transform=test_transform
        )
        generator = torch.Generator().manual_seed(42)
        self.train_set, _ = torch.utils.data.random_split(
            train_dataset, [400, 50], generator=generator
        )
        train_val_set, val_set = torch.utils.data.random_split(
            val_dataset, [400, 50], generator=generator
        )
        self.train_val_set = train_val_set
        self.val_set = val_set

        # Test set.
        self.test_set = PseudoGTPhone(
            self.root, train=False, transform=test_transform
        )

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
                self.train_val_set,
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

import os
import glob

from typing import Callable, Optional, Sequence
from argparse import ArgumentParser

# from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset, Dataset, DataLoader
from monai.data import list_data_collate
from monai.utils import set_determinism
from monai.transforms import (
    apply_transform,
    Randomizable,
    Compose,
    OneOf,
    EnsureChannelFirstDict,
    LoadImageDict,
    SpacingDict,
    OrientationDict,
    DivisiblePadDict,
    CropForegroundDict,
    ResizeDict,
    RandZoomDict,
    ZoomDict,
    RandRotateDict,
    HistogramNormalizeDict,
    ScaleIntensityDict,
    ScaleIntensityRangeDict,
    ToTensorDict,
)

from pytorch_lightning import LightningDataModule


class UnpairedDataset(CacheDataset, Randomizable):
    def __init__(
        self,
        keys: Sequence,
        data: Sequence,
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None,
        batch_size: int = 32,
        is_training: bool = True,
    ) -> None:
        self.keys = keys
        self.data = data
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.data))
        else:
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        for key, dataset in zip(self.keys, self.data):
            if self.is_training:
                rand_idx = self.R.randint(0, len(dataset))
                data[key] = dataset[rand_idx]
                data[f"{key}_idx"] = rand_idx
            else:
                data[key] = dataset[index]
                data[f"{key}_idx"] = index
            # Add the filename to the data dictionary
            data[f"{key}_pth"] = data[key]  # Assuming "image3d" contains the filename

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class UnpairedDataModule(LightningDataModule):
    def __init__(
        self,
        train_image2d_folders: str = "path/to/folder",
        val_image2d_folders: str = "path/to/folder",
        test_image2d_folders: str = "path/to/dir",
        train_samples: int = 1000,
        val_samples: int = 400,
        test_samples: int = 400,
        shape: int = 512,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        # self.setup()
        self.train_image2d_folders = train_image2d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image2d_folders = test_image2d_folders
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str = None, extension: str = "*.png"):
            assert folders is not None
            paths = [
                glob.glob(os.path.join(folder, extension), recursive=True)
                for folder in folders
            ]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files

        self.train_image2d_files = glob_files(
            folders=train_image2d_folders, extension="**/*.png"
        )

        self.val_image2d_files = glob_files(
            folders=val_image2d_folders, extension="**/*.png"
        )

        self.test_image2d_files = glob_files(
            folders=test_image2d_folders, extension="**/*.png"
        )

    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImageDict(keys=["image2d"]),
                EnsureChannelFirstDict(
                    keys=["image2d"],
                ),
                ScaleIntensityDict(
                    keys=["image2d"],
                    minv=0.0,
                    maxv=1.0,
                ),
                HistogramNormalizeDict(
                    keys=["image2d"],
                    min=0.0,
                    max=1.0,
                ),
                RandZoomDict(
                    keys=["image2d"],
                    prob=1.0,
                    min_zoom=0.95,
                    max_zoom=1.15,
                    padding_mode="constant",
                    mode=["area"],
                ),
                CropForegroundDict(
                    keys=["image2d"],
                    source_key="image2d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                ZoomDict(
                    keys=["image2d"], zoom=0.95, padding_mode="constant", mode=["area"]
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(
                    keys=["image2d"],
                ),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.train_image2d_files],
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.train_loader = DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImageDict(keys=["image2d"]),
                EnsureChannelFirstDict(
                    keys=["image2d"],
                ),
                ScaleIntensityDict(
                    keys=["image2d"],
                    minv=0.0,
                    maxv=1.0,
                ),
                HistogramNormalizeDict(
                    keys=["image2d"],
                    min=0.0,
                    max=1.0,
                ),
                CropForegroundDict(
                    keys=["image2d"],
                    source_key="image2d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                ZoomDict(
                    keys=["image2d"], zoom=0.95, padding_mode="constant", mode=["area"]
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(
                    keys=["image2d"],
                ),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.val_image2d_files],
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
            is_training=True,
        )

        self.val_loader = DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_transforms = Compose(
            [
                LoadImageDict(keys=["image2d"]),
                EnsureChannelFirstDict(
                    keys=["image2d"],
                ),
                ScaleIntensityDict(
                    keys=["image2d"],
                    minv=0.0,
                    maxv=1.0,
                ),
                HistogramNormalizeDict(
                    keys=["image2d"],
                    min=0.0,
                    max=1.0,
                ),
                CropForegroundDict(
                    keys=["image2d"],
                    source_key="image2d",
                    select_fn=(lambda x: x > 0),
                    margin=0,
                ),
                ZoomDict(
                    keys=["image2d"], zoom=0.95, padding_mode="constant", mode=["area"]
                ),
                ResizeDict(
                    keys=["image2d"],
                    spatial_size=self.shape,
                    size_mode="longest",
                    mode=["area"],
                ),
                DivisiblePadDict(
                    keys=["image2d"],
                    k=self.shape,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensorDict(
                    keys=["image2d"],
                ),
            ]
        )

        self.test_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.test_image2d_files],
            transform=self.test_transforms,
            length=self.test_samples,
            batch_size=self.batch_size,
            is_training=False,
        )

        self.test_loader = DataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return self.test_loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--datadir", type=str, default="data", help="data directory")
    parser.add_argument("--shape", type=int, default=512, help="isotropic img shape")
    parser.add_argument(
        "--vol_shape", type=int, default=256, help="isotropic vol shape"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size size")

    hparams = parser.parse_args()
    # Create data module

    train_image2d_folders = [
        os.path.join(
            hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"
        ),
    ]
    train_label2d_folders = []

    val_image2d_folders = [
        os.path.join(
            hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"
        ),
    ]

    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image2d_folders=train_image2d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image2d_folders=test_image2d_folders,
        shape=hparams.shape,
        vol_shape=hparams.vol_shape,
        batch_size=hparams.batch_size,
    )
    datamodule.setup(seed=hparams.seed)

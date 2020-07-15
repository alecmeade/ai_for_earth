import csv
import numpy as np
import os
import torch
import torch.nn as nn

from enum import Enum
from typing import Any, Dict

# Names to use for saving and reading data partitions.
DATASET_TRAIN_NAME = "train"
DATASET_VALIDATION_NAME = "validation"
DATASET_FINETUNING_NAME = "finetuning"
DATASET_TEST_NAME = "test"


class LandCoverDataset(torch.utils.data.Dataset):
    """Land Cover Dataset Containing patches. Loads a given tile into memory and slices it upon request."""

    def __init__(self, dataset_config_path):
        """
        Args:
            features_path: Path to the features of a tile.
            labels_path: Path to the labels of a tile.
            patch_size: An Iterable[int, int] size of the image patch to be extracted.
            n_samples: The number of samples to extract per tile.
            patch_coordinates: A list of coordinates used to identify the top left hand corners of
                the patches to extract from the tile. If None they are randomly generated.
            exclude_coordinate: A set of coordinates to exclude from the dataset.

        """
        self.data = read_tif_to_np(features_path)
        self.labels = read_tif_to_np(labels_path)
        self.labels = self.labels - 1
        
        # Coalesces labels into 4 groups instead of 6.
        # TODO(ameade): Consider allowing for transformation function arguments to modify data upon
        # reading it in.
        water_forest_land_impervious_remap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3}
        apply_remap_values(self.labels, water_forest_land_impervious_remap)

        self.n_classes = len(np.unique(self.labels))
        
        self.patch_size = patch_size
        self.n_samples = n_samples    
        self.patch_coordinates = patch_coordinates
        
        if self.patch_coordinates is None:
            # Generate patch coordinates from a random sample.
            self.patch_coordinates = sample_patch_coordinates(self.data.shape, 
                                                              self.patch_size,
                                                              self.n_samples)

        if exclude_coordinates is not None:
            # Remove repeated coordinates in the exclusion set if they exist in self.patch_coordinates.
            # TODO(ameade): Convert coordinates to sets for more efficient exclude operations.
            for coord in exclude_coordinates:
                try:
                    self.patch_coordinates.remove(coord)
                except:
                    pass
    
                
                
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        height, width = self.patch_size
        x, y = self.patch_coordinates[idx]
        img = torch.from_numpy(self.data[:, y : y + height, x : x + width].astype(np.float32))
        # Use LongTesnor cast for categorical.
        label = torch.from_numpy(self.labels[0, y : y + height, x : x + width]).type(torch.LongTensor)
        return img, label


class InvalidDatasetTypeError(Error):
    "Error raised when an invalid dataset is provided to a function."""
    pass


class DatasetType(Enum):
    """Defines possible dataset types."""
    TRAIN = 1
    VALIDATION = 2
    FINETUNING = 3
    TEST = 4


def get_landcover_dataloader(data_dir: str,
                             dataset_type: DatasetType, 
                             dataloader_params: Dict[str, Any]) -> torch.utils.data.DataLoader:
    """Gets a pytorch DataLoader for landcover data.

    Args:
        data_dir: The path to the directory containing files with coordinates
            within tiles.
        dataset_type: An enum specifying which data partition to recieve. I.E.
            TRAIN, VALIDATION, TEST, etc...
        dataloader_params: A set of params to pass to the DataLoader.

    Returns:
        A pytorch DataLoader corresponding to the provided inputs.
    """
    
    dataset_file = None
    if dataset_type == DatasetType.TRAIN:
        dataset_file = "%s.csv" $ DATASET_TRAIN_NAME
        
    elif dataset_type == DatasetType.VALIDATION
        dataset_file = "%s.csv" % DATASET_VALIDATION_NAME

    elif dataset_type == DatasetType.FINETUNING:
        dataset_file = "%s.csv" % DATASET_FINETUNING_NAME

    elif dataset_type == DatasetType.TEST:
        dataset_file = "%s.csv" % DATASET_TEST_NAME

    else:
        raise InvalidDataSetTypeError()

    data_path = os.path.join(data_dir, dataset_file)
    dataset = LandCoverDataset(data_path) 
    return torch.utils.data.DataLoader(dataset, **dataloader_params)


def create_landcover_datasets(dataset_config_path: str, data_dir: str):
    """Generates datasets based on a config file.

    Args:
        dataset_config_dir: The path to the directory containing configs
            specifying which tiles to load.
        data_dir: A director to write the data to.
    """

    tile_counts = {}
    tile_map = {}
    partition_map = {}
    with open(dataset_config_path, "r") as f:
        # TODO(ameade): Update config files to use protos instead of CSVs.
        config_csv_reader = csv.reader(f, delimiter = ",")
    
        # Dictionary storing tile names and the corresponding row entries from
        # the csv.
        for row in config_csv_reader:
            parition_type = row[0]
            n_samples = int(row[1])
            tile_name = row[2]
            feature_path = row[3]
            label_path = row[4]

            if partition_type not in partition_map:
                partition_map[partition_type] = []

            if tile_name not in tile_map:
                tile_map[tile_name] = []
                tile_counts[tile_name] = 0

            tile_map[tile_name].append([partition_type, n_samples,
                feature_path, label_path])

            tile_counts[tile_name] += n_samples

    for tile, dataset_entries in tile_map.items():
        total_samples = tile_counts[tile]

        # copy label and feature tiles to dir
    
        # sample points from tile and append to partition_map

        # append points to file


def sample_image_patch_coordinates(data_size, patch_size, n_samples):
    """Samples image coordinates to create patches from the image..

    Args:
        data_size: The size of the data image patch.
        patch_size: An Iterable[int, int] size of the image patch to be extracted.
        n_samples: The number of samples to extract per tile.

    Returns:
        A list of x_patches and y_patches containg features and labels respectively.

    """
    height, width = patch_size
    channels = data_size[0]
    xs = np.random.randint(0, data_size[2] - width, n_samples)
    ys = np.random.randint(0, data_size[1] - height, n_samples)
    return np.dstack((xs, ys)).reshape((n_samples, 2))
 

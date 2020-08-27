import csv
import dataset_utils as utils
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from collections import defaultdict 
from typing import Any, Dict, Iterable

# Directories names containing the landcover tiles and coordinates.
TILES_DIR = "tiles"
COORDINATES_DIR = "coordinates"

# File extensions for the x and y tile data.
TILE_X_EXT = "_x.npy"
TILE_Y_EXT = "_y.npy"

class MaxResampleCountExceeded(Exception):
    """An error thrown when an image patch cannot be sucessfully sampled."""

class LandCoverDataset(torch.utils.data.Dataset):
    """Land Cover Dataset Containing patches. Loads a given tile into memory and slices it upon request."""

    def __init__(self, dataset_dir: str, partition_type: utils.PartitionType):
        """
        Args:
            dataset_dir: The base directory of the dataset.
        """
         
        self.dataset_config = utils.get_dataset_config_path(dataset_dir)
        self.coords_dir = get_coordinates_dir(dataset_dir)
        self.coords_path = get_coordinates_partition_path(self.coords_dir, partition_type)
        self.tiles_dir = get_tiles_dir(dataset_dir)

        # Cache all coordinates for the provided partition.
        self.index = {}
        tiles = set() 
        with open(self.coords_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.index[i] = row
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                tile = row[4]
                self.index[i] = [x1, y1, x2, y2, tile]
                tiles.add(tile)

        patch_width = self.index[0][2] - self.index[0][0]
        patch_height = self.index[0][3] - self.index[0][1]
        self.patch_size = (patch_height, patch_width)
        self.n_samples = len(self.index)
               
        # Calculate the number of classes in the dataset.
        self.classes = set()
        for tile in tiles: 
            labels = np.load(get_tile_y_path(self.tiles_dir, tile))
            self.classes.update(np.unique(labels))
        
        self.n_classes = len(self.classes)
        
        # Create variables to stored the currently cached tile and populate it.
        self.tile = None
        self.tile_x = None
        self.tile_y = None
        self.load_tile(tiles.pop())

    def load_tile(self, tile: str):
        """Loads the x and y data for a given tile and caches the data.
        
        Args:
            tile: The name of the tile to retrieve. If the tile is already
                stored in cache it is not read again.

        """
        if self.tile is None or self.tile != tile:
            self.tile = tile
            self.tile_x = np.load(get_tile_x_path(self.tiles_dir, tile))
            self.tile_y = np.load(get_tile_y_path(self.tiles_dir, tile))


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Retrieves a single image patch and corresponding labels."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x1, y1, x2, y2, tile = self.index[idx]
        self.load_tile(tile)
        features = torch.from_numpy(self.tile_x[:, y1:y2, x1:x2].astype(np.float32))
        label = torch.from_numpy(self.tile_y[0, y1:y2, x1:x2]).type(torch.LongTensor)

        return features, label


def create_landcover_dataset_from_config(dataset_dir: str):
    """Generates datasets based on a config file.
    
    Args:
        dataset_dir: The path to the dataset directory containing a config
            from which to create a landcover dataset.
    """

    # Create tiles and coordinates directories.
    tiles_dir = get_tiles_dir(dataset_dir) 
    utils.mkdir_clean(tiles_dir)
    coords_dir = get_coordinates_dir(dataset_dir)
    utils.mkdir_clean(coords_dir)

    tile_partitions = {}
    dataset_config = utils.get_dataset_config_path(dataset_dir) 

    patch_size = [0, 0]
    
    # A dict for mapping label values.
    label_map = {}

    with open(dataset_config + ".csv", "r") as f:
        reader = csv.reader(f)
    
        # Dictionary storing tile names and the corresponding row entries from
        # the csv.
        for i, row in enumerate(reader):
            if row[0] == "patch_size":
                # Read patch size information from config.
                patch_height = int(row[1])
                patch_width = int(row[2])
                patch_size = [patch_height, patch_width] 

            elif row[0] == "label_map":
                # Read information on how to map labels in the config.
                old_label = int(row[1])
                new_label = int(row[2])
                label_map[old_label] = new_label

            elif row[0] == "tile":
                # Read the of tiles contained in each dataset partition.
                partition_type = utils.PartitionType(int(row[1]))
                n_samples = int(row[2])
                tile = row[3]
                feature_path = row[4]
                label_path = row[5]

                if tile not in tile_partitions:
                    # Add a new data partition for the current tile.
                    tile_partitions[tile] = [n_samples, 
                                             feature_path,
                                             label_path,
                                             [[partition_type, n_samples]]]

                else:
                    # If the tile already exists increase the total number of
                    # samples and track the new partition.
                    tile_partitions[tile][0] += n_samples
                    tile_partitions[tile][3].append([partition_type, n_samples])

            elif row[0].startswith("#"):
                # This is a comment row, skip
                pass

        for tile, entry in tile_partitions.items():
            total_samples, feature_path, label_path, partitions = entry

            # Read x and y features of tile.
            tile_x = utils.read_tif_to_np(feature_path)
            tile_y = utils.read_tif_to_np(label_path)
            unique_labels, counts = np.unique(tile_y, return_counts=True)
            assert 0 not in (unique_labels), ("%s contains unlabeled patches."
                % tile)
            
            utils.apply_remap_values(tile_y, label_map)

            # Save the tile x and y features to a local directory to avoid multiple
            # reads from blobstore.
            np.save(get_tile_x_path(tiles_dir, tile), tile_x)
            np.save(get_tile_y_path(tiles_dir, tile), tile_y)

            # Sample numerous patches from the provided tile and write them to
            # the corresponding partition files. The sampled coordinates
            # correspond to the upper left hand corner of the patch.
            
            existing_samples = {}
            sample_idx = 0
            for partition_entry in partitions:
                partition_type, n_samples = partition_entry 
                coords_path = get_coordinates_partition_path(coords_dir, partition_type)

                with open(coords_path, "a") as f:
                    writer = csv.writer(f)
                    for i in range(sample_idx, sample_idx + n_samples):
                        resample_count = 0
                        resample_max = 100
                        resample = True
                        while resample_count < resample_max and resample:
                            x, y = sample_image_patch(tile_x.shape, patch_size, 1)[0, :]
                            # Gets the upper left and bottom right coordinates
                            # of the patch and write to file.
                            x1 = x
                            x2 = x + patch_size[1]
                            y1 = y
                            y2 = y + patch_size[0]
                            
                            # Some NAIP data is blank with all zero channels. We
                            # resample patches to avoid this.
                            patch = tile_x[:, y1:y2, x1:x2]
                            num_zero_channels = np.sum(np.sum(patch==0, axis=0) == 4)

                            if (x, y) in existing_samples or num_zero_channels > 0: 
                                #or (num_zero_channels / np.product(patch_size)) > 0.05:
                                resample = True
                                resample_count += 1

                            else:
                                existing_samples[(x, y)] = 0
                                writer.writerow([x1, y1, x2, y2, tile])
                                resample = False

                        if resample_count == resample_max:
                            raise MaxResampleCountExceeded(tile)

                    sample_idx = sample_idx + n_samples

def get_tile_x_path(tile_dir: str, tile: str) -> str:
    """Gets the path to the x features of a given tile.

    Args:
        tile_dir: The directory containing the tile.
        tile: The name of the tile.

    Returns:
        The path to the tiles x feature data.

    """
    return os.path.join(tile_dir, tile + TILE_X_EXT)


def get_tile_y_path(tile_dir: str, tile: str) -> str:
    """Gets the path to the x features of a given tile.

    Args:
        tile_dir: The directory containing the tile.
        tile: The name of the tile.

    Returns:
        The path to the tiles x feature data.

    """
    return os.path.join(tile_dir, tile + TILE_Y_EXT)


def get_tiles_dir(dataset_dir: str) -> str:
    """Gets the directory of the tiles for a given landcover dataset.

    Args:
        dataset_dir: The lowest level dir for the dataset containing the config
        file.
    
    Returns:
        The path to the tiles directory.
    """
    return os.path.join(dataset_dir, TILES_DIR)


def get_coordinates_dir(dataset_dir: str) -> str:
    """Gets the directory of the coordinates for a given landcover dataset.

    Args:
        dataset_dir: The lowest level dir for the dataset containing the config
        file.
    
    Returns:
        The path to the tiles directory.
    """
    return os.path.join(dataset_dir, COORDINATES_DIR)


def get_coordinates_partition_path(coordinates_dir: str, 
                                   partition_type: utils.PartitionType) -> str:
    """Gets the path to the coordinates for the provided partition file.
    
    Args:
        coordinates_dir: A directory containing coordinates files.
        partition_type: A partition type to retrieve the coordinates for.

    Returns:
        The path to a given set of partition coordinates.
    """

    return os.path.join(coordinates_dir,
            utils.get_partition_name(partition_type) + ".csv")


def get_landcover_dataloader(dataset_dir: str,
                             partition_type: utils.PartitionType, 
                             dataloader_params: Dict[str, Any]) -> torch.utils.data.DataLoader:
    """Gets a pytorch DataLoader for landcover data.

    Args:
        dataset_dir: The path to the directory containing files with coordinates
            within tiles.
        partition_type: An enum specifying which data partition to recieve. I.E.
            TRAIN, VALIDATION, TEST, etc...
        dataloader_params: A set of params to pass to the DataLoader.

    Returns:
        A pytorch DataLoader corresponding to the provided inputs.
    """
    
    dataset = LandCoverDataset(dataset_dir, partition_type) 
    return torch.utils.data.DataLoader(dataset, **dataloader_params)

def get_landcover_dataloaders(dataset_dir: str,
                              partition_types: Iterable[utils.PartitionType],
                              dataloader_params: Dict[str, Any],
                              force_create_dataset: bool = True) -> Iterable[torch.utils.data.DataLoader]:
    """Gets a list pytorch DataLoaders for landcover data.

    Args:
        dataset_dir: The path to the directory containing files with coordinates
            within tiles.
        partition_types: An iterable of enums specifying which data partition to recieve. I.E.
            TRAIN, VALIDATION, TEST, etc...
        dataloader_params: A set of params to pass to the DataLoader.
        force_create_dataset: Whether or not to populate the dataset directory
            from the config in dataset_dir. If the dataset already exists it
            will be deleted and remade when True.

    Returns:
        An Iterable DataLoader for a LandcoverDataset for each partition.
    """
    
    if force_create_dataset:
        create_landcover_dataset_from_config(dataset_dir)

    return [get_landcover_dataloader(dataset_dir, p, dataloader_params) for
        p in partition_types]


def sample_image_patch(data_size, patch_size, n_samples):
    """Samples image coordinates to create patches from the image..

    Args:
        data_size: The size of the data image patch.
        patch_size: An Iterable[int, int] size of the image patch to be extracted.
        n_samples: The number of samples to extract per tile.

    Returns:
        A list of x_patches and y_patches containg features and labels respectively.

    """
    height, width = patch_size
    xs = np.random.randint(0, data_size[2] - width, n_samples)
    ys = np.random.randint(0, data_size[1] - height, n_samples)
    return np.dstack((xs, ys)).reshape((n_samples, 2))
 
def plot_prediction_overlay(tile: np.ndarray, prediction: np.ndarray):
    """Generates a plot of a landcover tile and its corresponding model
    predictions.
    
    Args:
        tile: The landcover tile to plot.
        prediction: The class label predictions of the landcover tile.

    """
    plt.figure()
    plt.imshow(tile)
    plt.show()



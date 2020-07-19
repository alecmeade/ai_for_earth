import numpy as np
import os 
import rasterio
import shutil

from enum import Enum
from typing import Dict

# Names to use for saving and reading data partitions.
PARTITION_TRAIN_NAME = "train"
PARTITION_VALIDATION_NAME = "validation"
PARTITION_FINETUNING_NAME = "finetuning"
PARTITION_TEST_NAME = "test"

# File name for the config file specifying how to create the dataset.
DATASET_CONFIG_NAME = "dataset_config"

class InvalidPartitionTypeError(Exception):
    "Error raised when an invalid dataset is provided to a function."""
    pass

class PartitionType(Enum):
    """Defines possible dataset types."""
    TRAIN = 0
    VALIDATION = 1
    FINETUNING = 2
    TEST = 3

def get_partition_name(partition_type: PartitionType) -> str:
    """Gets the name of the provided partition type.

    Args:
        partition_type: The partition name to get the name of.

    Returns:
        The name corresponding to the partition type.

    """
    if partition_type == PartitionType.TRAIN:
        return  PARTITION_TRAIN_NAME
        
    elif partition_type == PartitionType.VALIDATION:
        return PARTITION_VALIDATION_NAME

    elif partition_type == PartitionType.FINETUNING:
        return PARTITION_FINETUNING_NAME

    elif partition_type == PartitionType.TEST:
        return PARTITION_TEST_NAME

    else:
        raise InvalidDatasetTypeError()

def get_dataset_config_path(dataset_dir: str) -> str:
    """Gets the path to the dataset config file for a given dataset. 

    Args:
        dataset_dir: The lowest level dir for the dataset containing the config file.
    
    Returns:
        The path to the tiles directory.
    """
    return os.path.join(dataset_dir, DATASET_CONFIG_NAME)


def read_tif_to_np(tif_path: str):
    """Reads a tif file and converts it into a numpy.ndarray.
    
    Arg:
        tif_path: The full path to the tif file to read.
    
    Returns:
        A numpy.ndarray containing the tif file data. The returned tif has a rolled
        dimension and so the input is in the shape (channels, height width).
    
    """
    with rasterio.open(tif_path) as f:
        return f.read()

def apply_remap_values(labels: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    """Reassigns values inplace in an numpy array given a provided mapping.
    
    Args:
        labels: An ndarray of labels.
        label_map: A dict[int, int] mapping label classes [original, new].
        
    """
    for l1, l2 in label_map.items():
        labels[labels == l1] = l2
 
def mkdir_clean(dir_path: str):
    """Makes an empty directory and clears all files within any existing dir of
    the same name.

    Args:
        dir_path: The path to the directory to clear and create.

    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)



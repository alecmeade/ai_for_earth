import numpy as np
import os
import torch
import torch.nn as nn


class LandCoverDataset(torch.utils.data.Dataset):
    """Land Cover Dataset Containing patches. Loads a given tile into memory and slices it upon request."""

    def __init__(self, features_path, labels_path, patch_size, n_samples, 
                 patch_coordinates = None, exclude_coordinates = None):
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

def read_tif_to_np(tif_path):
    """Reads a tif file and converts it into a numpy.ndarray.
    
    Arg:
        tif_path: The full path to the tif file to read.
    
    Returns:
        A numpy.ndarray containing the tif file data. The returned tif has a rolled
        dimension and so the input is in the shape (channels, height width).
    
    """
    with rasterio.open(tif_path) as f:
        return f.read()

def apply_remap_values(labels, label_map):
    """Reassigns values inplace in an numpy array given a provided mapping.
    
    Args:
        labels: An ndarray of labels.
        label_map: A dict[int, int] mapping label classes [original, new].
        
    """
    for l1, l2 in label_map.items():
        labels[labels == l1] = l2

def sample_patch_coordinates(data_size, patch_size, n_samples):
    """Generates image patches from a tile containing features and corresponding labels.

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
    


import numpy as np
import rasterio

from typing import Dict

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


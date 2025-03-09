import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import h5py
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastFieldReader:
    """
    Standalone fast reader for Field HDF5 diagnostic files.
    """
    def __init__(self, file_path: str, field_names: Optional[List[str]] = None):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        self.h5file = h5py.File(self.file_path, 'r')

        # Load data items without loading full data into memory
        self.data_items = {}
        try:
            for name, item in self.h5file["data"].items():
                self.data_items[int(name[4:])] = item

            # Sort items by timestep
            self.data_items = dict(sorted(self.data_items.items()))
            first_item = next(iter(self.data_items.values()))

            # Set available fields
            if field_names is None:
                self.available_fields = list(first_item.keys())
            else:
                self.available_fields = [f for f in field_names if f in first_item]

            # Get field properties from the first field
            first_field = first_item[self.available_fields[0]]
            self.shape = first_field.shape
            self.offset = first_field.attrs.get('gridGlobalOffset', np.zeros(len(self.shape)))
            self.spacing = first_field.attrs.get('gridSpacing', np.ones(len(self.shape)))

            # Precompute axes coordinates (they are constant for the file)
            self.axes_coords = self.get_axes()

            # Check for cylindrical fields and, if so, determine modes
            self.is_cylindrical = any("_mode_" in field for field in self.available_fields)
            self.modes = {}
            if self.is_cylindrical:
                for field in self.available_fields:
                    if "_mode_" in field:
                        base_field, mode = field.rsplit("_mode_", 1)
                        if base_field not in self.modes:
                            self.modes[base_field] = []
                        self.modes[base_field].append(int(mode))
                for field in self.modes:
                    self.modes[field].sort()

        except Exception as ex:
            self.h5file.close()
            raise Exception(f"Error initializing FastFieldReader: {str(ex)}")
        else:
            logger.info(f"Field loaded successfully.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        logger.info(f"Field closed successfully.")

    def close(self):
        if hasattr(self, 'h5file') and self.h5file:
            self.h5file.close()

    @property
    def timestamps(self) -> np.ndarray:
        try:
            times = [str(a.name[4:]) for a in self.h5file["data"].values()]
            return np.array(times, dtype=np.str_)
        except Exception as ex:
            logger.exception(f"Unexpected error extracting timestamps: {ex}")
            return np.array([], dtype=np.float64)

    def get_axes(self) -> List[np.ndarray]:
        axes = []
        for i in range(len(self.shape)):
            axis = np.arange(
                self.offset[i],
                self.offset[i] + (self.shape[i] - 0.5) * self.spacing[i],
                self.spacing[i]
            )
            axes.append(axis)
        return axes

    def get_available_fields(self) -> List[str]:
        return list(self.modes.keys()) if self.is_cylindrical else self.available_fields

    def get_info(self) -> Dict[str, Any]:
        info = {
            "file_path": str(self.file_path),
            "shape": self.shape,
            "offset": self.offset,
            "spacing": self.spacing,
            "timestamps": self.timestamps,
            "is_cylindrical": self.is_cylindrical,
            "available_fields": self.get_available_fields(),
        }
        if self.is_cylindrical:
            info["modes"] = self.modes
        logger.info(f"Path: {info['file_path']}")
        logger.info(f"Shape: {info['shape']}")
        logger.info(f"Spacing: {info['spacing']}")
        logger.info(f"Found {len(info['timestamps'])} timestamps ({info['timestamps'][0]} - {info['timestamps'][-1]})")
        logger.info(f"Available fields: {info['available_fields']}")
        return info

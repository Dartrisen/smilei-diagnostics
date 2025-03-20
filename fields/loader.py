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

    def get_field_at_time(self, field: str, timestamp: str, subset: Optional[Dict[str, slice]] = None) -> np.ndarray:
        # Find closest timestep if necessary
        if timestamp not in self.timestamps:
            idx = np.argmin(np.abs(self.timestamps.astype(np.float64) - float(timestamp)))
            timestamp = self.timestamps[idx]

        # Get item for timestep
        t_idx = list(self.data_items.keys())[list(self.timestamps).index(timestamp)]
        item = self.data_items[t_idx]

        # Prepare selection
        if subset is None:
            selection = tuple(slice(None) for _ in range(len(self.shape)))
        else:
            selection = tuple(subset.get(str(i), slice(None)) for i in range(len(self.shape)))

        # Simple field case
        if field in item:
            data = np.empty(self.shape)
            item[field].read_direct(data, source_sel=selection)
            return data

        # Handle cylindrical fields with modes
        if self.is_cylindrical and field in self.modes:
            if subset and "theta" in subset:
                theta = subset["theta"]
                data = np.zeros(self.shape)
                for mode in self.modes[field]:
                    mode_field = f"{field}_mode_{mode}"
                    if mode_field not in item:
                        continue
                    real_sel = list(selection)
                    if len(real_sel) > 1:
                        if isinstance(real_sel[1], slice):
                            real_sel[1] = slice(
                                None if real_sel[1].start is None else real_sel[1].start * 2,
                                None if real_sel[1].stop is None else real_sel[1].stop * 2,
                                (real_sel[1].step or 1) * 2
                            )
                        else:
                            real_sel[1] = real_sel[1] * 2
                    real_sel = tuple(real_sel)
                    real_data = np.empty(self.shape)
                    item[mode_field].read_direct(real_data, source_sel=real_sel)
                    data += np.cos(mode * theta) * real_data
                    if mode > 0:
                        imag_sel = list(selection)
                        if len(imag_sel) > 1:
                            if isinstance(imag_sel[1], slice):
                                imag_sel[1] = slice(
                                    (imag_sel[1].start or 0) * 2 + 1,
                                    None if imag_sel[1].stop is None else imag_sel[1].stop * 2 + 1,
                                    (imag_sel[1].step or 1) * 2
                                )
                            else:
                                imag_sel[1] = imag_sel[1] * 2 + 1
                        imag_sel = tuple(imag_sel)
                        imag_data = np.empty(self.shape)
                        item[mode_field].read_direct(imag_data, source_sel=imag_sel)
                        data += np.sin(mode * theta) * imag_data
                return data
            else:
                mode_field = f"{field}_mode_0"
                if mode_field not in item:
                    raise ValueError(f"Field {field} mode 0 not found")
                data = np.empty(self.shape)
                item[mode_field].read_direct(data, source_sel=selection)
                return data

        raise ValueError(f"Field {field} not found")

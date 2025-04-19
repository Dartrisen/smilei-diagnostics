import logging
import os
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
        self.timesteps = self._get_timesteps()

        # Load data items without loading full data into memory
        self.data_items = {}
        try:
            for name, item in self.h5file["data"].items():
                # Use os.path.basename to get the numeric part, e.g. "0000003426"
                basename = os.path.basename(name)
                timestep = int(basename)
                self.data_items[timestep] = item

            # Sort items by timestep (keys are integers)
            self.data_items = dict(sorted(self.data_items.items()))
            first_item = next(iter(self.data_items.values()))
            # Set available fields
            if field_names is None:
                self.available_fields = list(first_item.keys())
            else:
                self.available_fields = [f for f in field_names if f in first_item]

            # Use the shape of the first field as the default shape.
            first_field = first_item[self.available_fields[0]]
            self.data_shape = first_field.shape
            self.offset = first_field.attrs.get('gridGlobalOffset', np.zeros(len(self.data_shape)))
            self.spacing = first_field.attrs.get('gridSpacing', np.ones(len(self.data_shape)))

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

        except Exception as e:
            self.h5file.close()
            raise Exception(f"Error initializing FastFieldReader: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, 'h5file') and self.h5file:
            self.h5file.close()

    def _get_timesteps(self) -> np.ndarray:
        try:
            # Extract the basename (e.g. "0000003426") from each group name
            times = [int(os.path.basename(a.name)) for a in self.h5file["data"].values()]
            return np.array(times, dtype=np.int64)
        except Exception as ex:
            print(ex)
            return np.array([], dtype=np.float64)

    def get_axes(self) -> List[np.ndarray]:
        axes = []
        for i in range(len(self.data_shape)):
            axis = np.arange(
                self.offset[i],
                self.offset[i] + (self.data_shape[i] - 0.5) * self.spacing[i],
                self.spacing[i]
            )
            axes.append(axis)
        return axes

    def get_field_at_time(self, field: str, timestep: Any,
                          subset: Optional[Dict[Any, slice]] = None) -> np.ndarray:
        """
        Get the field data at a given timestep.

        Parameters:
            field   : name of the field (e.g. "Rho_electron")
            timestep: can be a number or string that identifies the desired timestep.
            subset  : dictionary where keys are integer indices for dimensions and
                      an optional key "theta" to specify the angle (for cylindrical fields).
                      For example: {0: slice(10,100), 1: slice(20,200), "theta": 0.5}
        """
        # Convert the input timestep into an integer and find the closest available one
        try:
            t_val = int(float(timestep))
        except Exception as e:
            raise ValueError("Invalid timestep provided") from e

        if t_val not in self.data_items:
            # Find the closest timestep
            available = np.array(list(self.data_items.keys()))
            idx = np.argmin(np.abs(available - t_val))
            t_val = int(available[idx])
        item = self.data_items[t_val]

        if subset is None:
            subset = {}

        # Separate out "theta" if present in the subset since it is not a slicing key
        theta = None
        if subset is not None and "theta" in subset:
            theta = subset["theta"]
            subset = {k: v for k, v in subset.items() if k != "theta"}

        selection = tuple(subset.get(i, slice(None)) for i in range(len(self.data_shape)))

        if field in item:
            current_shape = item[field].shape
            data = np.empty(current_shape)
            item[field].read_direct(data, source_sel=selection)
            return data

        if self.is_cylindrical and field in self.modes:
            if theta is not None:
                sample_mode_field = f"{field}_mode_{self.modes[field][0]}"
                if sample_mode_field not in item:
                    raise ValueError(f"Field {field} mode {self.modes[field][0]} not found")
                current_shape = item[sample_mode_field].shape
                data = np.zeros(current_shape)
                for mode in self.modes[field]:
                    mode_field = f"{field}_mode_{mode}"
                    if mode_field not in item:
                        continue
                    real_sel = list(selection)
                    if len(real_sel) > 1:
                        if isinstance(real_sel[1], slice):
                            start = None if real_sel[1].start is None else real_sel[1].start * 2
                            stop = None if real_sel[1].stop is None else real_sel[1].stop * 2
                            step = (real_sel[1].step or 1) * 2
                            real_sel[1] = slice(start, stop, step)
                        else:
                            real_sel[1] = real_sel[1] * 2
                    real_sel = tuple(real_sel)
                    real_data = np.empty(current_shape)
                    item[mode_field].read_direct(real_data, source_sel=real_sel)
                    data += np.cos(mode * theta) * real_data

                    if mode > 0:
                        imag_sel = list(selection)
                        if len(imag_sel) > 1:
                            if isinstance(imag_sel[1], slice):
                                start = (imag_sel[1].start or 0) * 2 + 1
                                stop = None if imag_sel[1].stop is None else imag_sel[1].stop * 2 + 1
                                step = (imag_sel[1].step or 1) * 2
                                imag_sel[1] = slice(start, stop, step)
                            else:
                                imag_sel[1] = imag_sel[1] * 2 + 1
                        imag_sel = tuple(imag_sel)
                        imag_data = np.empty(current_shape)
                        item[mode_field].read_direct(imag_data, source_sel=imag_sel)
                        data += np.sin(mode * theta) * imag_data
                return data
            else:
                mode_field = f"{field}_mode_0"
                if mode_field not in item:
                    raise ValueError(f"Field {field} mode 0 not found")
                current_shape = item[mode_field].shape
                data = np.empty(current_shape)
                item[mode_field].read_direct(data, source_sel=selection)
                return data

        raise ValueError(f"Field {field} not found in timestep {t_val}")

    def get_available_fields(self) -> List[str]:
        return list(self.modes.keys()) if self.is_cylindrical else self.available_fields

    def get_info(self) -> Dict[str, Any]:
        info = {
            "file_path": str(self.file_path),
            "shape": self.data_shape,
            "offset": self.offset,
            "spacing": self.spacing,
            "timesteps": self.timesteps,
            "is_cylindrical": self.is_cylindrical,
            "available_fields": self.get_available_fields(),
        }
        if self.is_cylindrical:
            info["modes"] = self.modes
        return info

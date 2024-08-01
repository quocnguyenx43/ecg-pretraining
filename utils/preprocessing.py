import torch
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt


class PadSequence:
    def __init__(self, target_len_sec: int = 10) -> None:
        self.target_len_sec = target_len_sec  
    def __call__(self, x: np.ndarray, fs: int) -> np.ndarray:
        length_sec = x.shape[1] / fs
        target_samples = self.target_len_sec * fs
        if length_sec < self.target_len_sec:
            pad_width = target_samples - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
        else:
            x = x[:, :target_samples]
        return x

class Resample:
    def __init__(self, target_len: int = None, target_fs: int = None) -> None:
        self.target_length = target_len
        self.target_fs = target_fs
    def __call__(self, x: np.ndarray, fs: int = None) -> np.ndarray:
        if fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x
    
class RandomCrop:
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x.")
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]

class SOSFilter:
    def __init__(self, fs: int, cutoff: float, order: int = 5, btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')
    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class HighpassFilter(SOSFilter):
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')

class LowpassFilter(SOSFilter):
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')
        
class Standardize:
    def __init__(self, axis) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis
    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        return np.divide(x - loc, scale, out=np.zeros_like(x), where=scale != 0)

class ToTensor:
    _DTYPES = {"float": torch.float32, "double": torch.float64, "int": torch.int32, "long": torch.int64,}
    def __init__(self, dtype=torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype
    def __call__(self, x) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)
    
class Compose:
    def __init__(self, transforms) -> None:
        self.transforms = transforms
    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x
    
PREPROCESSING = {
    'pad_sequence': PadSequence, 
    'resample': Resample,
    'random_crop': RandomCrop,
    'highpass_filter': HighpassFilter,
    'lowpass_filter': LowpassFilter,
    'standardize': Standardize,
}
    
def get_data_preprocessor(config):
    transforms = []
    for transform in config:
        name, kwargs = list(transform.items())[0]
        transforms.append(PREPROCESSING[name](**kwargs))
    return transforms
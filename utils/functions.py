import torch
import numpy as np
from scipy.io import loadmat


# load ecg and ecg header
def load_data(filename: str):
    x = loadmat(filename + '.mat')
    data = np.asarray(x['val'], dtype=np.float64)
    with open(filename + '.hea', 'r') as f:
        header_data = f.readlines()
    return data, header_data

def setup_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
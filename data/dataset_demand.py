import os
import sys
import pdb
import json
import numpy as np
from six.moves import cPickle
from tqdm import tqdm
import warnings
import librosa
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


class NoiseDEMANDPathSet(Dataset):
    """
    Dataset of paths of noise files (DEMAND, .wav files)
    self.fpath_list     : (list) Path (str) to .wav files
    """
    def __init__(self, json_path, root_dir):
      
        # Load path list
        with open(json_path, 'r') as f:
            json_list = json.load(f)
        # Add root directory in front of files
        self.path_list = []
        for f in json_list:
            self.path_list.append(os.path.join(root_dir, f))
       
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        return self.path_list[idx]

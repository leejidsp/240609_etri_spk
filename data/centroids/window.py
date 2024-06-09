import os
import sys
import math
import scipy.signal as sig
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd



def extract_windows(data, frame_size, overlap_size, window_type, drop_last=False):

    # Check
    if overlap_size == 0 and window_type != 'none':
        raise ValueError("To use a window, overlap_size should not be zero.")

    # Define windows
    shift_size = frame_size - overlap_size
    
    if overlap_size != 0:
        if window_type == 'none':
            overlap_func = np.ones(overlap_size*2)
        elif window_type == 'sine':
            overlap_func = sig.cosine(overlap_size*2)
            
        window_func = np.concatenate([overlap_func[:overlap_size], 
                                        np.ones(frame_size - overlap_size*2),
                                        overlap_func[overlap_size:]])
    else:
        window_func = np.ones(frame_size)

        
    # Extract windows
    num_windows = int(math.ceil(float(len(data))/shift_size))
    if drop_last:
        num_windows -= 1

    windows = []
    for i in range(0, num_windows):
        # Get the frame
        iBgn = shift_size * i
        iEnd = iBgn + frame_size
        frame = data[iBgn:iEnd]

        # Modify length of data
        if drop_last == True and (i+1) == num_windows:
            data = data[:iEnd]

        # Pad frame if len(frame) < frame_size
        if drop_last == False and len(frame) < frame_size:
            num_pad = frame_size - len(frame)
            frame = np.pad(frame, (0, num_pad), 
                                    'constant', constant_values=[0])
            data = np.pad(data, (0, num_pad),
                                    'constant', constant_values=[0])
        
        win_frame = frame * window_func

        # Add each frame
        if i == 0:
            windows = np.reshape(np.array(win_frame), (1, frame_size))
        else:
            windows = np.append(windows, np.reshape(np.array(win_frame), (1, frame_size)), axis=0)
            
    if windows.shape[0]*windows.shape[1] != len(data):
        print("{} and {}".format(windows.shape[0]*windows.shape[1], len(data)))

    # windows: np.array (#frame, frame_size)
    # data: np.array (T_pad=#frame*frame_size)
    return windows, data





# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of spectral operations."""
import librosa
import numpy as np
import scipy.signal.windows as W
import scipy
# import tensorflow.compat.v2 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-8  # Small constant to avoid division by zero.

# Mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

### This is a 'translation' from Google's tf implementation to the torch edition

def torch_aligned_random_crop(waves, frame_length):
    """Get aligned random crops from batches of input waves."""
    n, t = waves[0].shape
    # n: batch
    # t: T if d=0 or T-1 if d=1
    
    crop_t = frame_length * (t//frame_length - 1)
    offsets = [torch.randint(size=(),low=0,high=t-crop_t,dtype=torch.int32)
               for _ in range(n)]

    waves_unbatched = [list(torch.split(w, 1, dim=0)) for w in waves]

    wave_crops = [[torch.narrow(torch.narrow(w,0,0,1),1,start=o,length=crop_t)
                   for w, o in zip(ws, offsets)] for ws in waves_unbatched]

    wave_crops = [torch.cat(wc, dim=0) for wc in wave_crops]

    return wave_crops


def torch_mel_to_hertz(frequencies_mel):
    """Converts frequencies in `frequencies_mel` from mel to Hertz scale."""
    # return _MEL_BREAK_FREQUENCY_HERTZ * (
    #         tf.math.exp(frequencies_mel / _MEL_HIGH_FREQUENCY_Q) - 1.)
    return _MEL_BREAK_FREQUENCY_HERTZ * (
            torch.exp(torch.tensor(frequencies_mel) / _MEL_HIGH_FREQUENCY_Q) - 1.)


def torch_hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    # return _MEL_HIGH_FREQUENCY_Q * tf.math.log(
    #     1. + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))
    return _MEL_HIGH_FREQUENCY_Q * torch.log(
        1. + (torch.tensor(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))


def torch_get_spectral_matrix(n, num_spec_bins=256, use_mel_scale=True,
                        sample_rate=16000):
    """DFT matrix in overcomplete basis returned as a TF tensor.
    Args:
      n: Int. Frame length for the spectral matrix.
      num_spec_bins: Int. Number of bins to use in the spectrogram
      use_mel_scale: Bool. Equally spaced on Mel-scale or Hertz-scale?
      sample_rate: Int. Sample rate of the waveform audio.
    Returns:
      Constructed spectral matrix.
    """
    sample_rate = float(sample_rate)
    upper_edge_hertz = sample_rate / 2.
    lower_edge_hertz = sample_rate / n

    if use_mel_scale:
        upper_edge_mel = torch_hertz_to_mel(upper_edge_hertz)
        lower_edge_mel = torch_hertz_to_mel(lower_edge_hertz)
        mel_frequencies = torch.linspace(lower_edge_mel, upper_edge_mel, num_spec_bins)
        hertz_frequencies = torch_mel_to_hertz(mel_frequencies)
    else:
        hertz_frequencies = torch.linspace(lower_edge_hertz, upper_edge_hertz, num_spec_bins)

    time_col_vec = (torch.reshape(torch.arange(n, dtype=torch.float32), [n, 1])
                    * np.cast[np.float32](2. * np.pi / sample_rate))

    tmat = torch.reshape(hertz_frequencies, [1, num_spec_bins]) * time_col_vec
    dct_mat = torch.cos(tmat)
    dst_mat = torch.sin(tmat)

    dft_mat = torch.view_as_complex(torch.stack([dct_mat,-dst_mat], dim=-1))

    return dft_mat


def torch_matmul_real_with_complex(real_input, complex_matrix):
    real_part = torch.matmul(real_input, torch.view_as_real(complex_matrix)[:, :, 0])
    imag_part = torch.matmul(real_input, torch.view_as_real(complex_matrix)[:, :, 1])

    return torch.view_as_complex(torch.stack([real_part, imag_part], dim=-1))


def torch_calc_spectrograms(waves, window_lengths, spectral_diffs=(0, 1),
                      window_name='hann', use_mel_scale=True,
                      proj_method='matmul', num_spec_bins=256,
                      sampling_rate=16000,
                      random_crop=True):
    """Calculate spectrograms with multiple window sizes for list of input waves.
    Args:
      waves: List of float tensors of shape [batch, length] or [batch, length, 1].
      window_lengths: List of Int. Window sizes (frame lengths) to use for
        computing the spectrograms.
      spectral_diffs: Int. order of finite diff. to take before computing specs.
      window_name: Str. Name of the window to use when computing the spectrograms.
        Supports 'hann' and None.
      use_mel_scale: Bool. Whether or not to project to mel-scale frequencies.
      proj_method: Str. Spectral projection method implementation to use.
        Supported are 'fft' and 'matmul'.
      num_spec_bins: Int. Number of bins in the spectrogram.
      random_crop: Bool. Take random crop or not.
    Returns:
      Tuple of lists of magnitude spectrograms, with output[i][j] being the
        spectrogram for input wave i, computed for window length j.
    """
    # w: (B, 1, T)
    waves = [torch.squeeze(w, dim=1) for w in waves]
    # w: (B, T)

    if window_name == 'hann':
        windows = [torch.reshape(torch.hann_window(wl, periodic=False).cuda(), (1, 1, -1))
                    for wl in window_lengths]
        # window: (1, 1, -1)
    elif window_name is None:
        windows = [None] * len(window_lengths)
    else:
        raise ValueError('Unknown window function (%s).' % window_name)

    spec_len_wave = []
    for d in spectral_diffs:
        for length, window in zip(window_lengths, windows):

            wave_crops = waves
            for _ in range(d):  # skip for d=0
                wave_crops = [w[:, 1:] - w[:, :-1] for w in wave_crops]

            if random_crop:
                wave_crops = torch_aligned_random_crop(wave_crops, length)
                #wave_crops: list of (B, crop_t)

            frames = [wc.unfold(-1, length, length//2) for wc in wave_crops]

            # TODO: Whether this method is feasible (in the gradient part) remains to be verified
            if window is not None:
                frames = [f * window for f in frames]

            if proj_method == 'fft':
                ffts = [torch.fft.rfft(f)[:, :, 1:] for f in frames]
            elif proj_method == 'matmul':
                mat = torch_get_spectral_matrix(length, num_spec_bins=num_spec_bins,
                                          use_mel_scale=use_mel_scale,
                                          sample_rate=sampling_rate)
                mat = mat.cuda()
                ffts = [torch_matmul_real_with_complex(f, mat) for f in frames]

            sq_mag = lambda x: (torch.view_as_real(x)[..., 0])**2 + (torch.view_as_real(x)[..., 1])**2

            specs_sq = [sq_mag(f) for f in ffts]

            if use_mel_scale and proj_method == 'fft':
                upper_edge_hertz = sampling_rate / 2.
                lower_edge_hertz = sampling_rate / length
                lin_to_mel = torchaudio.functional.melscale_fbanks(
                    n_freqs=length // 2 + 1, 
                    f_min=lower_edge_hertz,
                    f_max=upper_edge_hertz,
                    n_mels=num_spec_bins, 
                    sample_rate=sampling_rate
                    )[1:]
            
                specs_sq = [torch.matmul(s, lin_to_mel) for s in specs_sq]
            
            specs = [torch.sqrt(s+EPSILON) for s in specs_sq]
            spec_len_wave.append(specs)

    spec_wave_len = zip(*spec_len_wave)
    return spec_wave_len


def torch_sum_spectral_dist(specs1, specs2, add_log_l2=True):
    """Sum over distances in frequency space for different window sizes.
    Args:
      specs1: List of float tensors of shape [batch, frames, frequencies].
        Spectrograms of the first wave to compute the distance for.
      specs2: List of float tensors of shape [batch, frames, frequencies].
        Spectrograms of the second wave to compute the distance for.
      add_log_l2: Bool. Whether or not to add L2 in log space to L1 distances.
    Returns:
      Tensor of shape [batch] with sum of L1 distances over input spectrograms.
    """
    # specs1[i]: (B, #frame, #freq)
    l1_distances = [torch.mean(abs(s1 - s2), dim=[1, 2])
                    for s1, s2 in zip(specs1, specs2)]
    #l1_distances[i]: (B, )
    sum_dist = torch.sum(torch.stack(l1_distances),dim=0)   #sum over s=2**i's
    # sum_dist: (B, )

    if add_log_l2:
        log_deltas = [(torch.log(s1 + EPSILON)-torch.log(s2 + EPSILON))**2
            for s1, s2 in zip(specs1, specs2)]
        #log_deltas[i]: (B, #frame, #freq)
        log_l2_norms = [torch.mean(torch.sqrt(torch.mean(ld, dim=-1) + EPSILON), dim=-1)
            for ld in log_deltas]
        #log_l2_norms[i]: (B, )
        sum_log_l2 = torch.sum(torch.stack(log_l2_norms),dim=0)

        sum_dist += sum_log_l2

    return sum_dist



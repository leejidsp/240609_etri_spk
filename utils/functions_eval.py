from mir_eval.separation import bss_eval_sources
from pesq import pesq
#from pam1.jnd import *
import scipy.signal as signal

from utils.frame_runner import FrameRunner
from utils.utils import AttrDict

def calculate_evaluation_metric(eval_metric, desired, recon, sr, **kwargs):

    if eval_metric == 'SDR':
        result = bss_eval_sources(desired, recon, False)[0][0]
    elif eval_metric == 'PESQ':
        result = pesq(sr, desired, recon, 'wb')
    elif eval_metric == 'rnmr':
        result = cal_rnmr(desired, recon, sr)

    return result


def cal_rnmr(sig, sig_hat, sr, analysis_n_fft=1024):
    
    PN=90.302
    eps_spc = 1e-16
    n_fft = analysis_n_fft
    hann_win = signal.hann(n_fft)
    #hann_win = get_win(n_fft, type='hann', to_tensor=False)

    # frame_runner    
    h = AttrDict({'frame_size': n_fft, 'overlap_size': 0})
    frame_runner = FrameRunner(h, 'none', 'none')
    
    # Calculate jnd
    sig_frames, _ = frame_runner.extract_windows(sig)
    sig_hat_frames, _ = frame_runner.extract_windows(sig)
   
    _, jnd = fast_multi_frame_jnd(sig_frames, sr=sr, n_fft=n_fft)

    # PSD of reconstruction noise
    recon_noise = sig_hat - sig
    recon_noise_frames, _ = frame_runner.extract_windows(recon_noise)
    recon_noise_win = recon_noise_frames * hann_win

    #recon_noise_win = np.reshape(recon_noise, (-1, n_fft)) * hann_win
    recon_noise_fft = np.fft.rfft(recon_noise_win / n_fft, n=n_fft, axis=1)
    recon_noise_psd = PN + 10 * np.log10(np.absolute(recon_noise_fft) ** 2 + eps_spc)
    
    # Analysis
    zeros = np.zeros_like(jnd)
    recon_noise_vs_jnd = recon_noise_psd - jnd
    recon_noise_over_jnd = np.maximum(recon_noise_vs_jnd, zeros)
    rnmr = np.mean(recon_noise_over_jnd)

    return rnmr



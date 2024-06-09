import math
import scipy.signal as sig
import numpy as np
from scipy.io import wavfile
import torch



class FrameRunner(object):
    def __init__(self, h):
        super(FrameRunner, self).__init__()

        # Set parameters
        self.frame_size = h.frame_size
        self.overlap_size = int(h.overlap_ratio*h.frame_size)
        self.shift_size = self.frame_size - self.overlap_size
        window_in = h.window_in
        window_out = h.window_out

        # Check parameters
        if self.overlap_size == 0 and (window_in != 'none' or window_out != 'none'):
            raise ValueError("To use a window, overlap_size should not be zero.")

        # Define window (INPUT)
        if self.overlap_size != 0:
            if window_in == 'none':
                overlap_func_in = np.ones(self.overlap_size*2)
            elif window_in == 'sine':
                overlap_func_in = sig.cosine(self.overlap_size*2)
            elif window_in == 'hann':
                overlap_func_in = sig.hann(self.overlap_size*2)
            else:
                raise NotImplementedError("Not implemented window_in type.")
            self.window_func_in = np.concatenate([overlap_func_in[:self.overlap_size], 
                                        np.ones(self.frame_size - self.overlap_size*2),
                                        overlap_func_in[self.overlap_size:]])
        else:
            self.window_func_in = np.ones(self.frame_size)
        self.window_func_in_torch = torch.Tensor(self.window_func_in).cuda()    
        
        # Define window (OUTPUT)
        if self.overlap_size != 0:
            if window_out == 'none':
                overlap_func_out = np.ones(self.overlap_size*2)
            elif window_out == 'sine':
                overlap_func_out = sig.cosine(self.overlap_size*2)
            elif window_out == 'hann':
                overlap_func_out = sig.hann(self.overlap_size*2)
            else:
                raise NotImplementedError("Not implemented window_out type.")
            self.window_func_out = np.concatenate([overlap_func_out[:self.overlap_size], 
                                        np.ones(self.frame_size - self.overlap_size*2),
                                        overlap_func_out[self.overlap_size:]])
        else:
            self.window_func_out = np.ones(self.frame_size)
        self.window_func_out_torch = torch.Tensor(self.window_func_out).cuda()


    def extract_windows(self, data, drop_last=False):
        """ Extract frames from one file """
        # data: np.array (T) 

        # Extract windows
        num_windows = int(math.ceil(float(len(data))/self.shift_size))
        if drop_last:
            num_windows -= 1

        windows = []
        for i in range(0, num_windows):
            # Get the frame
            iBgn = self.shift_size * i
            iEnd = iBgn + self.frame_size
            frame = data[iBgn:iEnd]

            if drop_last == True and (i+1) == num_windows:
                data = data[:iEnd]

            # Pad frame if len(frame) < frame_size
            if drop_last == False and len(frame) < self.frame_size:
                num_pad = self.frame_size - len(frame)
                frame = np.pad(frame, (0, num_pad),
                                        'constant', constant_values=[0])
                data = np.pad(data, (0, num_pad),
                                        'constant', constant_values=[0])

            win_frame = frame * self.window_func_in
            # Add each frame
            if i == 0:
                windows = np.reshape(np.array(win_frame), (1, self.frame_size))
            else:
                windows = np.append(windows, np.reshape(np.array(win_frame), (1, self.frame_size)), axis=0)

        return windows, data  # np.array (#frame, frame_size)


    def reconstruct_windows(self, windows):
        """ Overlap-and-add frames into one file (numpy) """
        # windows : np.array (#frame, frame_size)

        for i in range(0, windows.shape[0]):
            r = windows[i, :]   # This frame
            r *= self.window_func_out   

            # Overlap and add
            if i == 0:
                recon = r
            else:
                if self.overlap_size != 0:
                    overlap_last = recon[-self.overlap_size:]
                    overlap_this = r[:self.overlap_size]
                    unmodified = r[self.overlap_size:]

                    overlapped = overlap_last + overlap_this

                    recon[-self.overlap_size:] = overlapped
                else:
                    unmodified = r
                recon = np.concatenate([recon, unmodified])

        # (T) 
        return recon
    

    def reconstruct_windows_batch(self, windows_batch):
        """ Overlap-and-add frames (in batch) into files (in batch) """
        # windows_batch: np.array (B, #frame, frame_size)   B: #files

        assert windows_batch.shape[0] != 1

        for i in range(0, windows_batch.shape[1]):
            r = windows_batch[:, i, :]      # (B, frame_size)
            r *= self.window_func_out

            # Overlap and add
            if i == 0:
                recon_mult = r  # (B, frame_size)
            else:
                if self.overlap_size != 0:
                    overlap_last = recon_mult[:, -self.overlap_size:]
                    overlap_this = r[:, :self.overlap_size]
                    unmodified = r[:, self.overlap_size:]
                    
                    overlapped = overlap_last + overlap_this    # (B, overlap_size)
    
                    recon_mult[:, -self.overlap_size:] = overlapped
                else:
                    unmodified = r

                recon_mult = np.concatenate([recon_mult, unmodified], axis=-1)
                
        # (B, T)
        return recon_mult


    def reconstruct_windows_torch(self, windows):
        """ Overlap-and-add frames into one file (torch) """
        # windows : torch.Tensor (#frame, frame_size)

        for i in range(0, windows.size(0)):
            r = windows[i, :]   # This frame
            r *= self.window_func_out_torch   

            # Overlap and add
            if i == 0:
                recon = r
            else:
                if self.overlap_size != 0:
                    overlap_last = recon[-self.overlap_size:]
                    overlap_this = r[:self.overlap_size]
                    unmodified = r[self.overlap_size:]

                    overlapped = overlap_last + overlap_this

                    recon[-self.overlap_size:] = overlapped
                else:
                    unmodified = r
                recon = torch.cat([recon, unmodified])    

        # (T)
        return recon
    

    def reconstruct_windows_batch_torch(self, windows_batch):
        """ Overlap-and-add frames (in batch) into files (in batch) (torch) """
        # windows_batch: torch.Tensor (B, #frame, frame_size)   B: #files

        assert windows_batch.size(0) != 1

        for i in range(0, windows_batch.size(1)):
            r = windows_batch[:, i, :]      # (B, frame_size)
            r *= self.window_func_out_torch

            # Overlap and add
            if i == 0:
                recon_mult = r  # (B, frame_size)
            else:
                if self.overlap_size != 0:
                    overlap_last = recon_mult[:, -self.overlap_size:]
                    overlap_this = r[:, :self.overlap_size]
                    unmodified = r[:, self.overlap_size:]
                    
                    overlapped = overlap_last + overlap_this    # (B, overlap_size)
    
                    recon_mult[:, -self.overlap_size:] = overlapped
                else:
                    unmodified = r

                recon_mult = torch.cat([recon_mult, unmodified], dim=-1)
                
        # (B, T)
        return recon_mult



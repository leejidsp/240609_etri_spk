# Import trainers
import numpy as np

class AttrDict(dict):
    """
    Make dictionary to attribute
    ex) input dictionary a = {'name': 'anonymous'}
        -> a.name == 'anonymous'
    """
    def __init__(self, *cfg, **kwcfg):
        super(AttrDict, self).__init__(*cfg, **kwcfg)
        self.__dict__ = self


#def entropy2bitrate(entropy, SR, STEP, EMBED):

#    return (SR / STEP) * EMBED * entropy / 1000

def entropy2bitrate(entropy, num_frames_per_sec, num_samples_per_frame):
    """
    Returns:
        bitrate: in kbps
    """
    bitrate =  num_frame_per_sec * num_samples_per_frame * entropy / 1000

    return bitrate

    
def wav_float2int(wav):

    wav = wav * 32768
    wav = np.clip(wav, -32768, 32767)
    wav = wav.astype(np.int16)

    return wav



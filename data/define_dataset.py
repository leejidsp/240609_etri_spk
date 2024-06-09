from .dataset_demand import NoiseDEMANDPathSet
from .dataset_libri import LibriTrainPathSet, LibriTrainFrameSet, LibriInferSet
#from .dataset_libri import CleanLibriTrainPathSet, CleanLibriTrainFrameSet, MixLibriDEMANDTrainFrameSetLibriInferSet

def load_dataset(mode, h, **kwargs):
    
    if mode == 'noise':
        return NoiseDEMANDPathSet(h.noise.json_path, h.root_dir)

    elif h.clean.data_name == 'LibriSpeech':
        # Train
        if mode == 'train_path':
            return LibriTrainPathSet(h.clean.json_path, h.root_dir)
        elif mode == 'train_frame':
            return LibriTrainFrameSet(kwargs['path_list'], kwargs['frame_runner'], h)

        # Infer
        elif mode.startswith('valid'):
            return LibriInferSet(mode, kwargs['num_file'], kwargs['frame_runner'], kwargs['seed'], h)

        elif mode.startswith('test'):
            return LibriInferSet(mode, kwargs['num_file'], kwargs['frame_runner'], h)





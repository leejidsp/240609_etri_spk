import os
import pdb
import random
import numpy as np
import json
import librosa
import argparse
from six.moves import cPickle
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from window import extract_windows


# Arguments
parser = argparse.ArgumentParser(description="Calculate wavlm centroids")

# Name
parser.add_argument('--encoder_layer',      default=6,          type=int)
parser.add_argument('--vocab_size',         default=128,        type=int)

# Waveform
parser.add_argument('--sampling_rate',      default=16000,      type=int)
parser.add_argument('--frame_sec',          default=1,          type=float)
parser.add_argument('--overlap_ratio',      default=0.,         type=float)
parser.add_argument('--drop_last',          default=1,          type=int)

# Kmeans params
parser.add_argument('--init',               default='k-means++',type=str)
parser.add_argument('--max_iter',           default=200,        type=int)
parser.add_argument('--batch_size',         default=10000,      type=int)
parser.add_argument('--tol',                default=0.0,        type=float)
parser.add_argument('--max_no_improvement', default=100,        type=int)
parser.add_argument('--n_init',             default=20,         type=int)
parser.add_argument('--reassignment_ratio', default=0.0,        type=float)

# Options
parser.add_argument('--file_batch_size',    default=500,        type=int)
parser.add_argument('--random_seed',        default=1111,       type=str)
parser.add_argument('--practice',           default=0,          type=int)
parser.add_argument('--save_dir',           default='centroids',type=str)
parser.add_argument('--data_root_dir',      default='/home/leeji/home1/dataset/LibriSpeech/', type=str)


# Calculate number of dataset
def create_data_path_list(data_dir):

    path_list = []
    for (root_dir, sub_dir, files) in os.walk(data_dir):
        for fname in files:
            ext = fname.split('.')[-1]
            if ext != 'flac':
                continue

            fpath = os.path.join(root_dir, fname)
            path_list.append(fpath)

    return path_list


def extract_wavlm(windows, wavlm_model, encoder_layer):
    
    # windows: (#frame, frame_size)
    
    for i in range(windows.shape[0]):
        sig = torch.Tensor(windows[i]).cuda()
        sig = F.pad(sig, (40, 40), "constant", 0)
        sig = sig.unsqueeze(0)
        # sig: (T,  )
        
        with torch.no_grad():   
            feature, _ = wavlm_model.extract_features(sig, output_layer=encoder_layer, ret_layer_results=False) # (1, feature_len, feature_dim)
        feature = feature.squeeze(0)    # (feature_len, feature_dim)
        feature = feature.detach().cpu().numpy()

        if i == 0:
            features = np.expand_dims(feature, 0)
        else:
            features = np.append(features, np.expand_dims(feature, 0), 0)
        
    return features 


def extract_features(fpath, frame_size, overlap_size, 
                            wavlm_model, encoder_layer, 
                            args):
    """
    Extract frames of each file and save as a pkl file
    """

    # Load waveform
    y, sr = librosa.load(fpath, sr=None, mono=False)

    # Extract windows
    windows, y = extract_windows(y, frame_size, overlap_size, 'none', args.drop_last)
    # windows: np.array (#frame, frame_size)

    # Extract wavlm features (raw, quantized)
    features = extract_wavlm(windows, wavlm_model, encoder_layer)
    # features: np.array (#frame, frame_size//downsample_factor, dimension)

    return features


class FileBatchDataset(Dataset):
    def __init__(self, path_list, i_batch, num_batch, model, 
                        frame_size, overlap_size, shift_size, args):

        pbar = tqdm(path_list, desc='FileBatch#{}/{}'.format(i_batch+1, num_batch), ncols=100)
        for i, fpath in enumerate(pbar):

            # Load frames of a pkl file
            features = extract_features(fpath, frame_size, overlap_size, 
                                        model, args.encoder_layer,
                                        args)
            
            if i == 0:
                feature_dim = features.shape[-1]
                stacked_features = features.reshape(-1, feature_dim)     # (feature_len, feature_dim)
            else:       
                stacked_features = np.append(stacked_features, features.reshape(-1, feature_dim), axis=0)

        self.stacked_features = stacked_features


    def __len__(self):
        return len(self.stacked_features)

    def __getitem__(self, idx):
        return self.stacked_features[idx] 


def main(args):

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    # [ Step 1 ] Prepare parameters and modules
    print("\n[ Step 1 ] Prepare parameters and modules")
    
    # Parameters
    frame_size = int(args.sampling_rate * args.frame_sec)
    overlap_size = int(frame_size * args.overlap_ratio)
    shift_size = frame_size - overlap_size

    # Prepare directory to save files
    keyword = 'wavlm_layer_{}_vocabsize_{}'.format(args.encoder_layer, args.vocab_size)
    #load_dir = os.path.join(args.save_dir, args.preprocess_name, 'train_'+keyword.replace('200', '100'))
    centroid_save_dir = os.path.join(args.save_dir, keyword)
    os.makedirs(centroid_save_dir, exist_ok=True)
    
    # Create km model
    km_model = MiniBatchKMeans(
                                n_clusters=args.vocab_size,
                                init=args.init,
                                max_iter=args.max_iter,
                                batch_size=args.batch_size,
                                verbose=1,
                                compute_labels=False,
                                tol=args.tol,
                                max_no_improvement=args.max_no_improvement,
                                init_size=None,
                                n_init=args.n_init,
                                reassignment_ratio=args.reassignment_ratio,
                                )


    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
    wavlm_model = knn_vc.wavlm.eval()


    # [ Step 2 ] Load data, infer model, and fit kmeans model
    print("\n[ Step 2 ] (train) Load and fit kmeans model")

    # Create data path list
    path_list = create_data_path_list(os.path.join(args.data_root_dir, 'train-clean-100'))
    num_files = len(path_list)
    assert num_files != 0, "Check if the data directory exists"

    path_list = sorted(path_list)
    np.random.shuffle(path_list)
    num_pathbatch = len(path_list) // args.file_batch_size + 1
    
    for i in range(num_pathbatch):
        iBgn = int(i*args.file_batch_size)
        iEnd = int((i+1)*args.file_batch_size)
        pathbatch = path_list[iBgn:iEnd]

        dataset = FileBatchDataset(path_list, i, num_pathbatch, wavlm_model, 
                                    frame_size, overlap_size, shift_size,
                                    args)

        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True)

        for j, batch in enumerate(dataloader):
            batch = batch.detach().numpy()
            if i==0 and j==0:
                km_model.fit(batch)
            else:
                km_model.partial_fit(batch)
            #print("FileBatch#{}-Batch#{}:\t".format(i, j), km_model.inertia_)
            #print(km_model.cluster_centers_[0, :10])


    # [ Step 3 ] (train) save quantized features
    print("\n[ Step 3 ] (train) save centroids")
    save_path = os.path.join(centroid_save_dir, 'centroids.pkl')
    if not args.practice:
        with open(save_path, 'wb') as cPickle_file:
            cPickle.dump(km_model.cluster_centers_, cPickle_file, protocol=cPickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':

    args = parser.parse_args()
    print()


    # Check each arguments
    print("[ Arguments ]")
    print("encoder_layer:   ", args.encoder_layer)
    print("vocab_size:      ", args.vocab_size)
    print()

    print("sampling_rate:   ", args.sampling_rate)
    print("frame_sec:       ", args.frame_sec)
    print("overlap_ratio:   ", args.overlap_ratio)
    print("drop_last:       ", bool(args.drop_last))
    print()

    print("init:            ", args.init)
    print("max_iter:        ", args.max_iter)
    print("batch_size:      ", args.batch_size)
    print("tol:             ", args.tol)
    print("max_no_improvement:", args.max_no_improvement)
    print("n_init:          ", args.n_init)
    print("reassignment_ratio:", args.reassignment_ratio)
    print()

    print("file_batch_size: ", args.file_batch_size)

    main(args)





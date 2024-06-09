import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import ResidualVectorQuantizer
from .hifigan_generator import GeneratorHiFi


class M02SameVQ(nn.Module):
    """
    """
    def __init__(self, h):
        super(M02SameVQ, self).__init__()

        # Set parameters
        self.feature_layer = h.FeatureExtractor.output_layer

        # Check configuration (h)
        self._check_hyperparameters(h)

        # Construct modules
        ### 0. Feature extractor
        knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
        self.feature_extractor= knn_vc.wavlm
        # TODO: freeze feature extractor depending on h.FeatureExtractor.freeze

        ### 1. Linguistic feature ----------
        # TODO

        ### 2. Speaker feature ----------
        # 2-1. Refiner 
        self.spk_refiner = Autoencoder(h.SpeakerRefiner)

        # 2-2. Quantizer
        h_spk_q = h.SpeakerQuantizer
        max_n_q, nbins = self._calculate_q_params(h_spk_q)
        print("SPK: max_n_q:{}, nbins:{} => {} kbps".format(max_n_q, nbins, h_spk_q.target_bandwidths[0]))
       
        self.spk_fr = h_spk_q.frame_rate
        self.spk_bw = h_spk_q.target_bandwidths[-1]

        self.spk_quantizer = ResidualVectorQuantizer(
                        dimension=h_spk_q.dimension,
                        n_q=max_n_q,
                        bins=nbins,
                        decay=h_spk_q.decay,
                        )


        # 4. HiFi-GAN
        self.vocoder = GeneratorHiFi(h.GeneratorHiFi, h.GeneratorHiFi.feature_dim)
        if h.GeneratorHiFi.pretrained_path != 'None':
            package = torch.load(h.GeneratorHiFi.pretrained_path, map_location='cpu')
        self.load_state_dict(package['vocoder'], strict=False)

        # Set gradient flow


    def _check_hyperparameters(self, h):
        pass
    

    def forward(self, x, quantize=[1, 1], centroid_dict=None, inference=False, **kwargs):
        assert quantize in [[1, 1], [1, 0]]

        # feature: (B, 1, T)
        feat_dict = {}
       
        ### 0. Extract feature
        x = x.squeeze(1)
        with torch.no_grad():    
            x_pad = F.pad(x, (40, 40), "constant", 0)
            feature, _ = self.feature_extractor.extract_features(x_pad, output_layer=self.feature_layer, ret_layer_results=False) # (B, feature_len, feature_dim)
            feature = feature.transpose(1, 2)   # (B, C, T)

        ### 1. Encode linguistic (Assign features to centroid)
        lin_raw = feature
        B, C, T = feature.size()

        lin_reshape = lin_raw.transpose(1, 2)   # (B, T, C)
        lin_reshape = lin_reshape.reshape(-1, C) # (B*T, C)

        # Calculate distances
        dist = ((lin_reshape**2).sum(1, keepdim=True)
                    - 2 * torch.matmul(lin_reshape, centroid_dict['centroid_t'])
                    + centroid_dict['c_t_norm']
                    )
        argindices = torch.argmin(dist, dim=1)

        lin_dec = centroid_dict['centroid'][argindices].reshape(B, T, C).transpose(1, 2)
        # (B, C, T)

        if 'utt_b' in kwargs.keys():
            spk_raw = kwargs['utt_b']
        else:
            spk_raw = feature - lin_dec

        ### 2. Encode speaker
        spk_enc = self.spk_refiner.encode(spk_raw)
        if not inference:
            spk_q_result = self.spk_quantizer.forward(spk_enc, sample_rate=self.spk_fr, bandwidth=self.spk_bw)
            q_spk_enc = spk_q_result.quantized
        else:
            spk_codes = self.spk_quantizer.encode(spk_enc, sample_rate=self.spk_fr, bandwidth=self.spk_bw)
            q_spk_enc = self.spk_quantizer.decode(spk_codes)
        spk_dec = self.spk_refiner.decode(q_spk_enc)
        feat_dict['spk_raw'] = spk_raw

        ### 4. Generate waveform 
        if quantize == [1, 1]:
            dec_feature = lin_dec + spk_dec
        elif quantize == [1, 0]:
            dec_feature = lin_dec
        waveform = self.vocoder(dec_feature)

        return waveform, feat_dict

    def inference_tag(self, x, q_tag, centroid_dict, **kwargs):

        if q_tag == 'W_LS':
            out, _ = self.forward(x, quantize=[1, 1], centroid_dict=centroid_dict, inference=True)
        if q_tag == 'W_L':
            out, _ = self.forward(x, quantize=[1, 0], centroid_dict=centroid_dict, inference=True)
        if q_tag == 'W_S':
            out1, _ = self.forward(x, quantize=[1, 1], centroid_dict=centroid_dict, inference=True)
            out2, _ = self.forward(x, quantize=[1, 0], centroid_dict=centroid_dict, inference=True)
            out = out1 - out2

        return out


    def _calculate_q_params(self, h):

        max_n_q = int(1000*h.target_bandwidths[-1] / (h.sampling_rate/h.hop_length*h.bits_per_stage))
        nbins = 2**(h.bits_per_stage)

        return max_n_q, nbins

# ============================================================================

class IdentityAE(nn.Module):
    def __init__(self,):
        super(IdentityAE, self).__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    """
    AutoEncoder of SoundStream (w/o vector quantization)
    """
    def __init__(self, h):
        """
        """
        super(Autoencoder, self).__init__()

        # Set parameters

        # Check configuration (h)
        self._check_hyperparameters(h)

        # Construct modules
        encoder_list = []
        for i in range(h.num_layers):
            in_ch = h.in_channels if i==0 else h.hidden_dimension
            out_ch = h.embed_channels if (i+1)==h.num_layers else h.hidden_dimension
            encoder_list.append(nn.Linear(in_ch, out_ch))
            if (i+1)!=h.num_layers:
                encoder_list.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*encoder_list)

        decoder_list = []
        for i in range(h.num_layers):
            in_ch = h.embed_channels if i==0 else h.hidden_dimension
            out_ch = h.in_channels if (i+1)==h.num_layers else h.hidden_dimension
            decoder_list.append(nn.Linear(in_ch, out_ch))
            if (i+1)!=h.num_layers:
                decoder_list.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*decoder_list)


    def _check_hyperparameters(self, h):
        pass        


    def encode(self, x):
        """
        Args:
            x       (torch.Tensor)  : input frame (B, C, T)
        Returns:
            z       (torch.Tensor)  : feature (B, D, T')
        """
        x = x.transpose(1, 2)
        z = self.encoder(x)
        z = z.transpose(1, 2)

        return z


    def decode(self, z):
        """
        Args
            z       (torch.Tensor)  : feature (B, D, T')
        Returns:
            x_dec   (torch.Tensor)  : decoded frame (B, C, T) 
        """
        z = z.transpose(1, 2)
        x_dec = self.decoder(z)
        x_dec = x_dec.transpose(1, 2)

        return x_dec
        

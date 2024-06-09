import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from .spectral_ops import torch_calc_spectrograms, torch_sum_spectral_dist


class LossReconMSDMPD(nn.Module):
    def __init__(self, loss_list, loss_weight, params, **kwargs):
        super(LossReconMSDMPD, self).__init__()

        # Set parameters
        self.loss_list = loss_list
        self.loss_weight = loss_weight
        self.loss_names_recon = []

        # Set pre-defined loss function
        self.lossfunc_l1 = nn.L1Loss()
        self.lossfunc_msmel = MultiScaleSpectralReconstructionLoss()


    def reset_loss_dict(self, ):

        self.losses_iter =  {'recon': {'weighted': {}, 'original': {}},
                            'adv': {'weighted': {}, 'original': {}}}
        self.losses_tb =    {'recon': {'weighted': {}, 'original': {}},
                            'adv': {'weighted': {}, 'original': {}},
                            'adv_sub': {}}


    def _update_loss_dict(self, name, loss_group):
        
        if name in ['msd_d_adv', 'mpd_d_adv']:
            loss, r_losses, g_losses = loss_group[0]
            w_loss = loss_group[1]

            self.losses_iter[name] = loss.item()

            self.losses_tb['adv']['original'][name] = loss.item()
            self.losses_tb['adv']['weighted'][name] = w_loss.item()
            
            for si in range(len(r_losses)):
                self.losses_tb['adv_sub'][name+'_sub_r'+str(si)] = r_losses[si]
                self.losses_tb['adv_sub'][name+'_sub_g'+str(si)] = g_losses[si]


        if name in ['msd_g_adv', 'mpd_g_adv']:
            loss, g_losses = loss_group[0]
            w_loss = loss_group[1]

            self.losses_iter[name] = loss.item()

            self.losses_tb['adv']['original'][name] = loss.item()
            self.losses_tb['adv']['weighted'][name] = w_loss.item()

            for si in range(len(g_losses)):
                self.losses_tb['adv_sub'][name+'_sub_g'+str(si)] = g_losses[si]


        elif name in ['msd_fm', 'mpd_fm']:
            loss, w_loss = loss_group

            self.losses_iter[name] = loss.item()

            self.losses_tb['adv']['original'][name] = loss.item()
            self.losses_tb['adv']['weighted'][name] = w_loss.item()


        elif name in ['recon_total', 'l1', 'msmel']:
            loss, w_loss = loss_group

            self.losses_iter[name] = loss.item()
            
            self.losses_tb['recon']['original'][name] = loss.item()
            self.losses_tb['recon']['weighted'][name] = w_loss.item()



# --- Dicsriminator ----------------------------------------------------------


    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


    def cal_loss_d(self, loss_msd_outputs, loss_mpd_outputs):

        # MSD
        loss_msd = loss_msd_outputs[0]
        w_loss_msd = loss_msd * self.loss_weight['msd_d_adv']

        # MPD
        loss_mpd = loss_mpd_outputs[0]
        w_loss_mpd = loss_mpd * self.loss_weight['mpd_d_adv']

        total_loss = w_loss_msd + w_loss_mpd

        # Update loss
        self._update_loss_dict('msd_d_adv', (loss_msd_outputs, w_loss_msd))
        self._update_loss_dict('mpd_d_adv', (loss_mpd_outputs, w_loss_mpd))

        return total_loss
    

# --- Generator --------------------------------------------------------------


    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
    
        return loss
    

    def generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l
    
        return loss, gen_losses


    def cal_loss_g(self, gen_x, real_x, loss_gen_msd_outputs, loss_gen_mpd_outputs, loss_fm_msd, loss_fm_mpd):
        
        # MSD adv
        loss_gen_msd = loss_gen_msd_outputs[0]
        w_loss_gen_msd = loss_gen_msd * self.loss_weight['msd_g_adv']

        # MPD adv
        loss_gen_mpd = loss_gen_mpd_outputs[0]
        w_loss_gen_mpd = loss_gen_mpd * self.loss_weight['mpd_g_adv']

        # MSD fm
        w_loss_fm_msd = loss_fm_msd * self.loss_weight['msd_fm']

        # MPD fm
        w_loss_fm_mpd = loss_fm_mpd * self.loss_weight['mpd_fm']

        # Reconstruction loss
        loss_l1 = self.lossfunc_l1(gen_x, real_x)
        w_loss_l1 = loss_l1 * self.loss_weight['l1']

        loss_msmel = self.lossfunc_msmel(gen_x, real_x)
        w_loss_msmel = loss_msmel * self.loss_weight['msmel']

        loss_recon = w_loss_l1 + w_loss_msmel
        w_loss_recon = loss_recon * self.loss_weight['recon_total']

        total_loss =    w_loss_gen_msd + w_loss_gen_mpd +\
                        w_loss_fm_msd + w_loss_fm_mpd +\
                        w_loss_recon

        # Update loss
        self._update_loss_dict('msd_g_adv', (loss_gen_msd_outputs, w_loss_gen_msd))
        self._update_loss_dict('mpd_g_adv', (loss_gen_mpd_outputs, w_loss_gen_mpd))
        self._update_loss_dict('msd_fm', (loss_fm_msd, w_loss_fm_msd))
        self._update_loss_dict('mpd_fm', (loss_fm_mpd, w_loss_fm_mpd))
        self._update_loss_dict('recon_total', (loss_recon, w_loss_recon))
        self._update_loss_dict('l1', (loss_l1, w_loss_l1))
        self._update_loss_dict('msmel', (loss_msmel, w_loss_msmel))

        return total_loss

# ============================================================================


class MultiScaleSpectralReconstructionLoss(nn.Module):
    def __init__(self, nbins=64, sr=16000, eps=1e-8, s_list=None):
        super(MultiScaleSpectralReconstructionLoss, self).__init__()

        self.eps = eps
        if s_list == None:
            self.s_list = [6, 7, 8, 9, 10, 11]
        else:
            self.s_list = s_list
        self.transform_list = [MelSpectrogram(sample_rate=sr, n_fft=2**i, hop_length=(2**i)//4, n_mels=64).cuda() for i in self.s_list]

    def forward(self, x_dec, x):

        loss = 0
        # loss = loss_l1 + loss_logl2
        for idx, i in enumerate(self.s_list):
            s = 2**i
            alpha_s = (s/2)**0.5
            
            mel_x       = self.transform_list[idx](x)             # (B, 1, #freq, #frames)
            mel_x_dec   = self.transform_list[idx](x_dec)
            
            mel_x       = mel_x.squeeze(1).transpose(1, 2)        # (B, #frames, #freq)
            mel_x_dec   = mel_x_dec.squeeze(1).transpose(1, 2)

            loss_s_mel = torch.mean((mel_x - mel_x_dec).abs(), dim=(1, 2))  # (B, )

            ld = (torch.log(mel_x+self.eps) - torch.log(mel_x_dec+self.eps))**2 # (B, #frames, #freq)
            loss_s_logmel = torch.mean(torch.sqrt(torch.mean(ld, dim=-1)+self.eps), dim=-1)     # (B, )

            loss_s = loss_s_mel + alpha_s * loss_s_logmel

            loss += loss_s

        # loss: (B, )
        loss = torch.mean(loss)     # scalar
        loss /= len(self.s_list)    # divide by the number of window lengths

        return loss



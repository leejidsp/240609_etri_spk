"""
Trainer - Sample
"""
### Basic libraries
import os
import sys
import pdb
import time
import math
import numpy as np
import shutil
import librosa
from tqdm import tqdm
from scipy.io import wavfile

### Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

### Custom
from data.define_dataset import load_dataset
from utils.utils import entropy2bitrate, wav_float2int
from utils.functions_plot import plot_spectrogram, plot_mel_spectrogram 
from utils.functions_eval import calculate_evaluation_metric


class TrainerHiFiAutoVCMean(object):
    def __init__(self, data_dict, model_dict, opt_dict, loss_function, cfg):
        """ Set parameters """

        # Global
        self.exp_name           = cfg.exp_name
        self.cfg                = cfg
        self.msd                = model_dict['MultiScaleDiscriminator']
        self.mpd                = model_dict['MultiPeriodDiscriminator']
        for key in model_dict.keys():
            if key.startswith('M0'):
                self.coder =  model_dict[key]
        self.opt_d              = opt_dict['Discriminator']
        self.opt_g              = opt_dict['Generator']
        self.lr_sched_d_dict    = cfg.opt['Discriminator'].lr_scheduler
        self.lr_sched_g_dict    = cfg.opt['Generator'].lr_scheduler
        self.loss_function      = loss_function

        # Data
        self.train_path_loader  = data_dict['train_path_loader']
        self.valid_loader_dict  = data_dict['valid_loader_dict']
        self.tb_loader_dict     = data_dict['tb_loader_dict']
        self.frame_runner       = data_dict['frame_runner']
        self.centroid_dict      = data_dict['centroid_dict']
        self.data               = cfg.data
        self.sampling_rate      = cfg.data.sampling_rate

        # Training
        self.n_epochs           = cfg.training.n_epochs
        self.batch_size         = cfg.training.batch_size

        # Train mode 
        self.use_resumption     = cfg.training.use_resumption           
        self.resumption_path    = cfg.training.resumption_path

        # Record
        self.q_wavs_eval        = cfg.record.q_wavs_eval
        self.eval_step          = cfg.record.eval_step
        self.eval_metrics       = cfg.record.eval_metrics
        self.tb_step            = cfg.record.tb_step
        self.tb_visualize_step  = cfg.record.tb_visualize_step
        self.tb_save_audio_step = cfg.record.tb_save_audio_step
        self.verbose_step       = cfg.record.verbose_step

        # File
        self.ckpt_step          = cfg.file.ckpt_step
        self.ckpt_dir           = cfg.file.ckpt_dir
        self.log_dir            = cfg.file.log_dir
        self.tb_dir             = cfg.file.tb_dir
        self.backup_dir         = cfg.file.backup_dir

        # Initialize
        self._initialize()



    def _initialize(self):
        """ Initialize or load training parameters """


        # --------------------------------------------------------------------
        print()

        # [RESUME] Load parameters of model to resume
        if self.use_resumption:
            pass
            #TODO: Modify according to new config hierarchy
            # Load and check parameters to resume
            #print("Load from checkpoint model: {}\n".format(self.resumption_path))
            #package = torch.load(self.resumption_path)
            #check_resumption_config(self.cfg, package['cfg'], 'ae')

            ## Load parameters for this run
            #self.autoencoder.load_state_dict(package['autoencoder_dict'])
            #self.optimizer_ae.load_state_dict(package['optimizer_ae_dict'])
            #self.start_epoch = package['epoch']
            #prev_lr_sched_ae_dict = package['cfg']['training'].lr_sched_ae_dict
            #self.lr_sched_ae_last_epoch = package['iteration'] // prev_lr_sched_ae_dict['step'] - 1
            #self.iteration = package['iteration']

            ## Restart log and summary from last iteration and epoch
            #self._restart_logsum(self.summary_path, self.start_epoch)
            #self._restart_logsum(self.log_path, self.iteration)


        # [Q-DISABLED or Q-SCRATCH] Start from scratch
        else:
            self.start_epoch = 0
            self.lr_sched_d_last_epoch = -1
            self.lr_sched_g_last_epoch = -1
            self.iteration = 0

        # --------------------------------------------------------------------

        # Set lr scheduler
        if self.lr_sched_d_dict.type == 'ExponentialLR':
            self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.opt_d,
                                                last_epoch=self.lr_sched_d_last_epoch,
                                                **self.lr_sched_d_dict.params)
        else:
            raise NotImplementedError("Not implemented learning rate scheduler.")

        if self.lr_sched_g_dict.type == 'ExponentialLR':
            self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.opt_g,
                                                last_epoch=self.lr_sched_g_last_epoch,
                                                **self.lr_sched_g_dict.params)
        else:
            raise NotImplementedError("Not implemented learning rate scheduler.")

        # --------------------------------------------------------------------

        # Define save directories
        self.save_dir_ckpt_iter     = self.ckpt_dir + self.exp_name + '/'

        # Create save directories
        os.makedirs(self.save_dir_ckpt_iter, exist_ok=True)

        # Define log & summary path and messages
        self.log_path = self.log_dir + self.exp_name + '.log'
        self.summary_path = self.log_dir + self.exp_name + '.summary'
        self._set_messages()
        
        # Create summarywriter/log/summary files
        self.tbsw = SummaryWriter(os.path.join(self.tb_dir, self.exp_name))
        if not self.use_resumption:
            with open(self.log_path, 'w') as f: 
                pass
            with open(self.summary_path, 'w') as f:
                pass

        # Create directories for valid samples
        self.save_dir_tb = os.path.join('eval_samples/'+self.data.mix.data_name+'_tb', self.exp_name)
        os.makedirs(self.save_dir_tb, exist_ok=True)

        # Backup for the first time
        if not self.use_resumption:
            os.system("./backup.sh {} train".format(self.exp_name)) 


    # ========================================================================
    #   MAIN
    # ========================================================================


    def train(self):
        """ Train and validate the model """

        print()
        print("Training...")
        for epoch in range(self.start_epoch, self.n_epochs):
            print()

            # ===== Train one epoch =====
            self.msd.train()
            self.mpd.train()
            self.coder.train()

            # Train
            st = time.time()
            tr_loss_epoch = self._train_one_epoch(epoch)
            train_time = time.time() - st
            # tr_loss_epoch:    dict (key: loss_name, value: loss_value)

            # ===== Record and save ===== 
            # Monitor
            print(self.monitor_train.format('Epoch: {:03d}'.format(epoch+1), **tr_loss_epoch))

            # Summary
            with open(self.summary_path, 'a') as f:
                f.write(self.line_summary_train.format(epoch+1, train_time=train_time, **tr_loss_epoch))


    # ========================================================================
    #   SUBMODULES FOR TRAINING (Train and Validation)
    # ========================================================================



    def _train_one_epoch(self, epoch):
        """ Train one epoch """

        losses = {ln: 0. for ln in self.loss_function.loss_list}
        iter_per_epoch = 0


        pbar = tqdm(self.train_path_loader, desc='Epoch {:03d}'.format(epoch+1), ncols=200)
        for path_list in pbar:
            # path_list: (B_path, )

            # ===== Define frame dataset =====
            train_frame_set = load_dataset('train_frame', self.data, 
                                            path_list=path_list, 
                                            frame_runner=self.frame_runner,
                                            )

            train_frame_loader = DataLoader(train_frame_set,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=True,
                                            num_workers=4)

            # ===== Main Part =====
            # For each filebatch
            for i, (x, p_x, _) in enumerate(train_frame_loader):
                # x :                     torch.FloatTensor   : input frames    # (B, T)
                real_x = x.unsqueeze(1).cuda()   # (B, 1, T)
                p_x = p_x.unsqueeze(1).cuda()
                
                real_x_loss = real_x.detach()   # for loss calculation

                """ (1) Inference through autoencoder """
                target = 'autovc_meanspk' if i%2==1 else 'default'
                if target.startswith('autovc'):
                    with torch.no_grad():
                        _, feat_dict = self.coder(p_x, [1, 1], self.centroid_dict)  # (B, 1, T)
                    if target.endswith('meanspk'):
                        utt_c = feat_dict['spk_enc_mean']
                        gen_x, feat_dict = self.coder(real_x, [1, 1], self.centroid_dict,
                                                    utt_c=utt_c)  # (B, 1, T)
                else:
                    gen_x, feat_dict = self.coder(real_x, [1, 1], self.centroid_dict)  # (B, 1, T)

                """ (2) Train discriminator """
                self.opt_d.zero_grad()

                self.loss_function.reset_loss_dict()
                # MSD
                d_msd_real, d_msd_fake, _, _ = self.msd(real_x_loss, gen_x.detach())
                loss_msd_outputs = self.loss_function.discriminator_loss(d_msd_real, d_msd_fake)

                # MPD
                d_mpd_real, d_mpd_fake, _, _ = self.mpd(real_x_loss, gen_x.detach())
                loss_mpd_outputs = self.loss_function.discriminator_loss(d_mpd_real, d_mpd_fake)

                # Total D loss and update
                loss_d = self.loss_function.cal_loss_d(loss_msd_outputs, loss_mpd_outputs)
                loss_d.backward()
                self.opt_d.step()


                """ (3) Train Generator """
                self.opt_g.zero_grad()

                # Infer through discriminator
                d_msd_real, d_msd_fake, fmap_msd_real, fmap_msd_fake = \
                                                        self.msd(real_x_loss, gen_x)

                d_mpd_real, d_mpd_fake, fmap_mpd_real, fmap_mpd_fake = \
                                                        self.mpd(real_x_loss, gen_x)

                # Feature loss
                loss_fm_msd = self.loss_function.feature_loss(fmap_msd_real, fmap_msd_fake)
                loss_fm_mpd = self.loss_function.feature_loss(fmap_mpd_real, fmap_mpd_fake)

                # Generator loss
                loss_gen_msd_outputs = self.loss_function.generator_loss(d_msd_fake)
                loss_gen_mpd_outputs = self.loss_function.generator_loss(d_mpd_fake)

                loss_g = self.loss_function.cal_loss_g(gen_x, real_x_loss,
                                                        loss_gen_msd_outputs, loss_gen_mpd_outputs,
                                                        loss_fm_msd, loss_fm_mpd)

                loss_g.backward()
                self.opt_g.step()

                losses_iter = self.loss_function.losses_iter 
                losses_tb = self.loss_function.losses_tb 


                """ (3) Return losses """
                # Stack losses
                for ln in losses.keys():
                    losses[ln] += losses_iter[ln]


                # ============================================================
                

                # Save model every ckpt_step
                if (self.iteration+1) % self.ckpt_step == 0:
                    filename = '{}_iteration_{:08d}.ckpt'.format(self.exp_name, self.iteration+1)
                    self._save_model(self.save_dir_ckpt_iter, filename, epoch, self.iteration)
            

                # Verbose training 
                if (self.iteration+1) % self.verbose_step == 0:
                    pbar.set_description(self.pbar_msg.format(epoch+1, self.iteration+1, losses_iter['recon_total']))


                # Record to tensorboard
                if (self.iteration+1) % self.tb_step == 0:
                    for key in losses_tb['recon']['weighted'].keys():
                        self.tbsw.add_scalar("0_1_0_train_w_loss_recon/"+key, losses_tb['recon']['weighted'][key], self.iteration+1)
                    for key in losses_tb['adv']['weighted'].keys():
                        self.tbsw.add_scalar("0_1_1_train_w_loss_adv/"+key, losses_tb['adv']['weighted'][key], self.iteration+1)
                    for key in losses_tb['recon']['original'].keys():
                        self.tbsw.add_scalar("0_2_0_train_loss_recon/"+key, losses_tb['recon']['original'][key], self.iteration+1)
                    for key in losses_tb['adv']['original'].keys():
                        self.tbsw.add_scalar("0_2_1_train_loss_adv/"+key, losses_tb['adv']['original'][key], self.iteration+1)
                    for key in losses_tb['adv_sub'].keys():
                        self.tbsw.add_scalar("0_2_1_train_loss_adv_sub/"+key, losses_tb['adv_sub'][key], self.iteration+1)


                # Visualize samples every tb_visualize_step iterations
                if self.tb_visualize_step != 0 and (self.iteration+1) % self.tb_visualize_step == 0:
                    self.coder.eval()
                    for snr in self.data.mix.snr_list:
                        key = str(snr)+'dB' if snr != 'clean' else 'clean'
                        tb_loader = self.tb_loader_dict[key]
                        for q_tag in self.q_wavs_eval:
                            self._visualize(self.batch_size, tb_loader, key, q_tag)
                    self.coder.train()


                # Validate every eval_step iterations
                if self.eval_step != 0 and (self.iteration+1) % self.eval_step == 0:
                    self.coder.eval()
                    val_metric_iter = self._validate(self.batch_size)
                    for em in self.eval_metrics:
                        self.tbsw.add_scalar("1_1_valid_metric/"+em, val_metric_iter[em], self.iteration+1)
                    print(self.monitor_valid.format('Iteration: {:08d}'.format(self.iteration+1), **val_metric_iter))                
                    self.coder.train()


                # Record to log file                    
                with open(self.log_path, 'a') as f:
                    f.write(self.line_log.format(self.iteration+1, **losses_iter))

                # Update lr scheduler
                if self.scheduler_d != None and (self.iteration+1) % self.lr_sched_d_dict.step == 0:
                    old_lr = self.scheduler_d.optimizer.param_groups[0]['lr']
                    self.scheduler_d.step()
                    new_lr = self.scheduler_d.optimizer.param_groups[0]['lr']
                    print("\nLearning rate changed from {} to {}\n".format(old_lr, new_lr))

                if self.scheduler_g != None and (self.iteration+1) % self.lr_sched_g_dict.step == 0:
                    old_lr = self.scheduler_g.optimizer.param_groups[0]['lr']
                    self.scheduler_g.step()
                    new_lr = self.scheduler_g.optimizer.param_groups[0]['lr']
                    print("\nLearning rate changed from {} to {}\n".format(old_lr, new_lr))

                # Increase counters
                self.iteration += 1
                iter_per_epoch += 1
                
                
        for ln in losses.keys():
            losses[ln] /= iter_per_epoch

        # losses:   dict {key: loss_name, value: loss_value}
        return losses



    @torch.no_grad()
    def _validate(self, num_file):
        """ Validate the model for SDR"""
        # num_file  : number of files to calculate at once (batch)
        
        eval_metric_mean = {em: 0. for em in self.eval_metrics}

        num_valid_wavs = len(self.valid_loader)
        pbar = tqdm(self.valid_loader, desc='Validation', ncols=180)
        for i, (_, full_frame, wavlen) in enumerate(pbar):
            # full_frame: (1, #frame, T)

            # Create list of files (size: num_file)
            if i % num_file == 0:
                full_frame_list = []
                wavlen_list = []
            full_frame_list.append(full_frame)
            wavlen_list.append(wavlen)

            # Infer when len(full_frame_list) == num_file or the end of pbar
            if (i+1) % num_file == 0 or (i+1) == len(pbar):
                # Infer multiple files at once
                desired_wav_list, recon_wav_list = self._infer_full_frame_multiple(full_frame_list, wavlen_list)

                # Calcualte evaluation metric
                for desired_wav, recon_wav in zip(desired_wav_list, recon_wav_list):
                    for em in self.eval_metrics:
                        eval_metric_mean[em] += calculate_evaluation_metric(em, desired_wav, recon_wav, self.sampling_rate)


             
        for em in self.eval_metrics:
            eval_metric_mean[em] /= num_valid_wavs

        # eval_metric_mean:     dict {key: loss_name, value: mean_loss_value}
        return eval_metric_mean



    @torch.no_grad()
    def _validate(self, num_file):
        """ Validate the model for SDR"""
        # num_file  : number of files to calculate at once (batch)
        
        eval_metric_mean = {em: 0. for em in self.eval_metrics}

        num_valid_wavs = len(self.valid_loader)
        pbar = tqdm(self.valid_loader, desc='Validation', ncols=180)
        for i, (_, f_frame, f_feature, f_q_feature, wavlen) in enumerate(pbar):
            # full_frame: (1, #frame, T)

            # Create list of files (size: num_file)
            if i % num_file == 0:
                f_frame_list = []
                f_feature_list = []
                f_q_feature_list = []
                wavlen_list = []
            f_frame_list.append(f_frame)
            f_feature_list.append(f_feature)
            f_q_feature_list.append(f_q_feature)
            wavlen_list.append(wavlen)

            # Infer when len(full_frame_list) == num_file or the end of pbar
            if (i+1) % num_file == 0 or (i+1) == len(pbar):
                # Infer multiple files at once
                desired_wav_list, recon_wav_list, _ = self._infer_frames_from_multiple_files(f_frame_list, f_feature_list, f_q_feature_list, wavlen_list)

                # Calcualte evaluation metric
                for desired_wav, recon_wav in zip(desired_wav_list, recon_wav_list):
                    for em in self.eval_metrics:
                        eval_metric_mean[em] += calculate_evaluation_metric(em, desired_wav, recon_wav, self.sampling_rate)


             
        for em in self.eval_metrics:
            eval_metric_mean[em] /= num_valid_wavs

        # eval_metric_mean:     dict {key: loss_name, value: mean_loss_value}
        return eval_metric_mean



    @torch.no_grad()
    def _visualize(self, num_file, tb_loader, snr_tag, q_tag):
        """ Visualize results of small samples """
    
        # Create directories for evaluation samples
        if (self.iteration+1) % self.tb_save_audio_step == 0:
            self.save_dir_tb_iter = os.path.join(self.save_dir_tb, 'iteration_{:08d}'.format(self.iteration+1))
            os.makedirs(self.save_dir_tb_iter+'/1_recon/'+snr_tag, exist_ok=True)

        desired_wav_list = []
        recon_wav_list = []
        # Create list of files (size: len(tb_loader))
        for i, (fname, x, wavlen) in enumerate(tb_loader):

            if i % num_file == 0:
                fname_list = []
                f_frame_list = []
                wavlen_list = []
            fname_list.append(fname[0])
            f_frame_list.append(x)
            wavlen_list.append(wavlen)

            if (i+1) % num_file == 0 or (i+1) == len(tb_loader):       
                desired_wav_l, recon_wav_l = self._infer_frames_from_multiple_files(f_frame_list, wavlen_list, q_tag)
                desired_wav_list += desired_wav_l
                recon_wav_list += recon_wav_l

        for i, (fname, desired_wav, recon_wav) in enumerate(zip(fname_list, desired_wav_list, recon_wav_list)):
            # Visualize desired wav only for the first time
            if (self.iteration+1) == self.tb_visualize_step:
                self.tbsw.add_audio('0_desired_{}/wav__{}__{}'.format(snr_tag, fname, q_tag), desired_wav, self.iteration+1, self.sampling_rate)
                self.tbsw.add_figure('0_desired_{}/spec__{}__{}'.format(snr_tag, fname, q_tag), plot_spectrogram(desired_wav, self.sampling_rate), self.iteration+1)
                
            # Visualize
            self.tbsw.add_audio('1_recon_{}/wav__{}__{}'.format(snr_tag, fname, q_tag), recon_wav, self.iteration+1, self.sampling_rate)

            # Add audio
            self.tbsw.add_figure('1_recon_{}/spec__{}__{}'.format(snr_tag, fname, q_tag), plot_spectrogram(recon_wav, self.sampling_rate), self.iteration+1)

            # Save samples
            if (self.iteration+1) % self.tb_save_audio_step == 0:
                wavfile.write(self.save_dir_tb_iter+'/1_recon/'+snr_tag+'/wav__{}__{}.wav'.format(fname, q_tag), 
                                        self.sampling_rate, wav_float2int(recon_wav))



    @torch.no_grad()
    def _infer_frames_from_multiple_files(self, f_frame_list, wavlen_list, q_tag):
        """ Infer full frames of multiple files """
    
        # frame_list: list of (1, #frame, T) torch.Tensor (list size=num_file=B)
        num_file = len(f_frame_list)
        
        """ (1) Prepare for inference """

        # Reconstruct desired waveform and search max number of frames
        desired_wav_list = []
        num_frame_list = []
        for f_frame, wavlen in zip(f_frame_list, wavlen_list):
            desired_wav = self.frame_runner.reconstruct_windows(f_frame.squeeze(0).numpy())
            desired_wav = desired_wav[:wavlen]
            desired_wav_list.append(desired_wav)
            num_frame_list.append(f_frame.size(1))
        
        # Pad each file with zeros to have save num_frame for all files
        max_num_frame = max(num_frame_list)
        for i, (f_frame, num_frame) in enumerate(zip(f_frame_list, num_frame_list)):
            # f_frame: (1, #frame, T)
            pad_len = max_num_frame - f_frame.size(1)
            f_frame_list[i] = F.pad(f_frame, pad=(0, 0, 0, pad_len, 0, 0), mode='constant', value=0)
            # padded f_frame: (1, max_num_frame, frame_len)
            assert f_frame_list[i].size(1) == max_num_frame
    
        """ (2) Infer """

        # Create batch of files and inference
        for fr_i in range(max_num_frame):
    
            """ (2-1) Stack input frames """
            # Stack a frame of each file 
            stacked_frame = torch.cat([f_frame_list[file_i][0, fr_i, :].unsqueeze(0) \
                                        for file_i in range(num_file)], dim=0)

            # stacked_frame: (B=num_file, T)
            stacked_frame = stacked_frame.unsqueeze(1).cuda()           # (B, 1, T)
    
            """ (2-2) Infer input frames """
            # Infer
            with torch.no_grad():
                stacked_gen_frame = self.coder.inference_tag(stacked_frame, q_tag, self.centroid_dict)
                # stacked_gen_frame: (B=num_file, 1, T) 

            """ (2-3) Stack output and intermediate frames """
            # Get output frames
            stacked_gen_frame_np = stacked_gen_frame.detach().cpu().numpy() # (B, 1, T)
            
            # Stack output frames
            if fr_i == 0:
                stacked_recon_frame = stacked_gen_frame_np.copy()  # (B, 1, T)
            else:
                stacked_recon_frame = np.append(stacked_recon_frame, stacked_gen_frame_np, axis=1)
            # stacked_recon_frame: (B, fr_i+1, T) => (B, max_num_frame, T)

   
        """ (3-1) Reconstruct waveforms"""
        # Reconstruct waveform after removing redundant (padded) frames
        recon_wav_list = []
        for i, (num_frames, wavlen) in enumerate(zip(num_frame_list, wavlen_list)):
            # Reconstructed signal
            recon_frame_i = stacked_recon_frame[i, :num_frames, :]   # (num_frames, T)
            recon_wav = self.frame_runner.reconstruct_windows(recon_frame_i)
            recon_wav = recon_wav[:wavlen]
            assert len(recon_wav) == len(desired_wav_list[i])
            recon_wav_list.append(recon_wav)

        return desired_wav_list, recon_wav_list



    # ========================================================================
    #   UTILS 
    # ========================================================================



    def _save_model(self, dir_path, fname, epoch, iteration):
        """ Save model parameters """

        save_path = os.path.join(dir_path, fname)
        torch.save({
                    'coder': self.coder.state_dict(),
                    'msd': self.msd.state_dict(),
                    'mpd': self.mpd.state_dict(),
                    'opt_d': self.opt_d.state_dict(),
                    'opt_g': self.opt_g.state_dict(),
                    'epoch': epoch+1,
                    'iteration': iteration+1,
                    'cfg': self.cfg,
                    },
                    save_path)


    # ========================================================================
    #   UTILS 
    # ========================================================================




    def _set_messages(self):
        """ Set messages to be used in monitor and logs """

        metric_msg = ' | '.join(["valid_"+em+": {"+em+":.4f}" for em in self.eval_metrics])
        loss_msg = ' | '.join([ln+": {"+ln+":.8f}" for ln in self.loss_function.loss_list])

        self.pbar_msg = "Epoch: {:04d} | Iteration: {:08d} | " + "recon_total: {:.4f}"
        
        self.monitor_train = "\n\t === [ train ] {:20s} ===================================================\n\t "\
                            + ' | ' + loss_msg\
                            +"\n\t ======================================================================================\n"

        self.monitor_valid = "\n\t === [ valid ] {:20s} ===================================================\n\t "\
                            + ' | ' + metric_msg\
                            +"\n\t ======================================================================================\n"

        self.line_log = "Iteration: {:08d} | " + loss_msg + "\n"

        self.line_summary_train = "Epoch: {:03d} | " + loss_msg +  " | time: {train_time:.0f} sec"




    def _restart_logsum(self, logsum_path, st_unit):

        f = open(logsum_path, 'r')
        previous_lines = f.readlines()

        # Find last iteration line
        for idx, line in enumerate(previous_lines):
            first_term = line.split("|")[0].split(":")[0]
            if first_term == 'Epoch' or first_term == 'Iteration':
                number = int(line.split("|")[0].split(":")[1])
                if number == (st_unit+1):
                    #print("restart: {} {} {}".format(number, st_unit, idx))
                    break
        f.close()

        if len(previous_lines) != 0:
            truncated_lines = previous_lines[:idx]
            if (idx+1) != len(previous_lines):
                open(logsum_path, 'w').writelines(truncated_lines)
        print("Restarted log/summary.")


    


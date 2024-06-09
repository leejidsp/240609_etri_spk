### Basic libraries
import os
import sys
import pdb
import argparse
import numpy as np
import json
import random
import yaml
import itertools
import hydra
from six.moves import cPickle

### Torch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

### Custom libraries
import models
from data.define_dataset import load_dataset
from losses.define_loss import define_loss
from trainers.define_trainer import define_trainer
from utils.frame_runner import FrameRunner

# GPU setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
print("Is GPU available?:\t", bool(torch.cuda.is_available()))
print("#Available GPU: \t\t", torch.cuda.device_count())


# Assign a config file name to CONFIG_NAME
parser = argparse.ArgumentParser(description="")
parser.add_argument('--config_name',
                        #default='example',
                        default='example2',
                        #default='00_m01_no_spk_refiner_s800',   # st9-0
                        #default='01_m01_use_spk_refiner_s800',  # st9-1

                            type=str, help="(str) Experiment name")

CONFIG_NAME = parser.parse_args().config_name

################
#     MAIN     #
################


@hydra.main(config_path='configs', config_name=CONFIG_NAME, version_base='1.2')
def main(cfg):
    
    assert cfg.exp_name==CONFIG_NAME, "Filename of a configuration file should be the same as exp_name."

    # Set necessary settings
    torch.set_num_threads(4)    # To avoid full use of CPU cores
    
    # Set random seeds
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # (Optional) Set CUDA and cuDNN for reproducibility
    #torch.backends.cuda.matmul.allow_t32 = False
    #torch.backends.cudnn.allow_tf32 = False
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    # Create basic directories
    os.makedirs(cfg.file.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.file.log_dir, exist_ok=True)
    os.makedirs(cfg.file.tb_dir, exist_ok=True)
    os.makedirs(cfg.file.backup_dir, exist_ok=True)


    # ===== MODEL =====
    print("Create model...")

    # Construct models
    model_dict = {}
    for module_name in cfg.model.module_list:
        model_dict[module_name] = models.all_model_dict[module_name](cfg.model[module_name]).cuda()
    #print(model_dict)

    print("... done.\n")


    # ===== DATA =====
    print("----------------------------------------------------------------\n")
    print("Set data...")

    # Define a FrameRunner module which extract and overlap-and-add frames
    frame_runner = FrameRunner(cfg.data.ola)

    # Define dataset and dataloader
    train_path_set = load_dataset('train_path', cfg.data)
    train_path_loader = DataLoader(train_path_set, batch_size=cfg.training.path_batch_size, shuffle=True)

    # Validation
    valid_set = load_dataset('valid', cfg.data, num_file=cfg.record.eval_num,
                                frame_runner=frame_runner)
    valid_loader_dict = {}
    for snr in cfg.data.mix.snr_list:
        key = str(snr)+'dB' if snr != 'clean' else 'clean'
        subset = valid_set.snr_subset_dict[key]
        valid_loader_dict[key] = DataLoader(subset, batch_size=1, shuffle=False)

    # Tensorboard
    tb_set = load_dataset('valid', cfg.data, num_file=cfg.record.tb_num,
                                frame_runner=frame_runner)
    tb_loader_dict = {}
    for snr in cfg.data.mix.snr_list:
        key = str(snr)+'dB' if snr != 'clean' else 'clean'
        subset = tb_set.snr_subset_dict[key]
        tb_loader_dict[key] = DataLoader(subset, batch_size=1, shuffle=False)

    # Load centroids
    centroid_path = os.path.join(cfg.data.centroid.root_dir,
                                'wavlm_layer_{}_vocabsize_{}.pkl'.format(
                                        cfg.data.centroid.layer, cfg.data.centroid.vocabsize))
    with open(centroid_path, 'rb') as cPickle_file:
        centroid = cPickle.load(cPickle_file)

    centroid_dict = {}
    centroid = torch.Tensor(centroid).cuda()
    centroid_t = centroid.transpose(0, 1)
    centroid_dict['centroid'] = centroid          # (#clusters, feature_dim)
    centroid_dict['centroid_t'] = centroid_t      # (feature_dim, #clusters)
    centroid_dict['c_t_norm'] = (centroid_t**2).sum(0, keepdim=True)   # (1, #clusters)
    
    
    # Print data statistics
    print("# of train files: ", len(train_path_loader))
    print()

    # Create data dictionary
    data_dict = {
            'train_path_loader': train_path_loader,
            'valid_loader_dict': valid_loader_dict,
            'tb_loader_dict': tb_loader_dict,
            'frame_runner': frame_runner,
            'centroid_dict': centroid_dict,
            }
    print("... done.\n")


    # ===== OPTIMIZER =====
    print("----------------------------------------------------------------\n")
    print("Set optimizers...")
    
    # Set optimizers
    opt_dict = {}
    for opt_key in cfg.opt.opt_group.keys():
        opt_value_list = cfg.opt.opt_group[opt_key]
        m_list = [model_dict[ov] for ov in opt_value_list]
        param_list = [m.parameters() for m in m_list]
        m_opt_dict = cfg.opt[opt_key].optimizer 

        if m_opt_dict.type == 'Adam':
            m_opt = torch.optim.Adam(itertools.chain(*param_list), **m_opt_dict.params)
        else:
            raise NotImplementedError("Not implemented optimizer.")
        opt_dict[opt_key] = m_opt


    # ===== LOSS FUNCTION =====
    print("----------------------------------------------------------------\n")
    print("Set a loss function...")

    loss_function = define_loss(**cfg.loss)


    # ===== RUN =====
    print("----------------------------------------------------------------\n")
    print("Activate Trainer...\n")

    trainer = define_trainer(cfg.training.trainer, data_dict, model_dict, opt_dict, loss_function, cfg)
    
    trainer.train()



###############
#     RUN     #
###############

if __name__ == '__main__':

    # Parse arguments
    print("----------------------------------------------------------------\n")
    print("config_name: ", CONFIG_NAME)
    print()


    # Check if there is the same experiment (based on tb file)
    tb_dir = os.path.join('tb/', CONFIG_NAME)
    if os.path.isdir(tb_dir):
        # Print messages
        print("The same experiment was trained before.\n" \
                "If you keep running this script, " \
                "previous results will be \033[31m overwritten. \033[0m \n" \
                "Are you sure to train again?\n")
        # TODO: activate if required
        #if cfg['train_mode'].use_resumption:
        #    print("\033[34m [RESUME] \033[0m The model will be trained from: \n" \
        #            "{}\n".format(cfg['train_mode'].resumption_path))

        # Get answer
        #answer = input("Write 'yes' if you want to keep running: ")
        #if answer != 'yes':
        #    print("\nStop running the experiment...\n")
        #    sys.exit()
        print()
    print("================================================================\n")

    # Run
    main()



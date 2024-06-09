# Import trainers
import numpy as np

def check_config(cfg, train_type=None):
    """
    Check if there is any collision between configurations
    """ 
    if train_type == 'ae':
        if len(cfg['network'].discriminators) == 0:
            assert cfg['training'].optimizer_d_dict['type'] == 'none'
            assert cfg['training'].lr_sched_d_dict['type'] == 'none'
    elif train_type == 'ldm':
        pass
    else:
        raise NotImplementedError()

    assert cfg['evaluation'].tb_save_samples_step >= cfg['evaluation'].tb_visualize_step
    assert cfg['evaluation'].tb_save_samples_step % cfg['evaluation'].tb_visualize_step == 0


# For autoencoders ===========================================================

def check_resumption_config(cfg, ckpt_cfg, train_type=None):
    """
    Check arguments are the same with pretrained model (same experiment)
    """
    assert train_type != None, NotImplementedError()

    assert cfg['exp_name'] == ckpt_cfg['exp_name']

    # Data
    assert cfg['data'].dataset_name == ckpt_cfg['data'].dataset_name
    assert cfg['data'].dataset_path == ckpt_cfg['data'].dataset_path
    assert cfg['data'].sampling_rate == ckpt_cfg['data'].sampling_rate
    assert cfg['data'].window_in == ckpt_cfg['data'].window_in
    assert cfg['data'].window_out == ckpt_cfg['data'].window_out

    # Network
    assert cfg['network'].autoencoder == ckpt_cfg['network'].autoencoder
    assert cfg['network'].autoencoder_config == ckpt_cfg['network'].autoencoder_config
    if train_type == 'ldm':
        assert cfg['network'].pretrained_ae_path == ckpt_cfg['network'].pretrained_ae_path
        assert cfg['network'].ldm == ckpt_cfg['network'].ldm
        assert cfg['network'].ldm_config == ckpt_cfg['network'].ldm_config
    assert cfg['network'].loss == ckpt_cfg['network'].loss
    assert cfg['network'].loss_ratio == ckpt_cfg['network'].loss_ratio

    # Training
    print("Pass cfg.n_epochs")
    if train_type == 'ae':
        assert cfg['training'].optimizer_ae_dict['lr'] == ckpt_cfg['training'].optimizer_ae_dict['lr']
        assert cfg['training'].lr_sched_ae_dict['type'] == ckpt_cfg['training'].lr_sched_ae_dict['type']
        assert cfg['training'].lr_sched_d_dict['type'] == ckpt_cfg['training'].lr_sched_d_dict['type']
    elif train_type == 'ldm':
        assert cfg['training'].optimizer_dict['lr'] == ckpt_cfg['training'].optimizer_dict['lr']
        assert cfg['training'].lr_sched_dict['type'] == ckpt_cfg['training'].lr_sched_dict['type']
    assert cfg['training'].path_batch_size == ckpt_cfg['training'].path_batch_size
    assert cfg['training'].batch_size == ckpt_cfg['training'].batch_size
    assert cfg['training'].seed == ckpt_cfg['training'].seed

    # Mode
    if train_type == 'ae':
        assert cfg['train_mode'].use_pretrained_ae == ckpt_cfg['train_mode'].use_pretrained_ae


def check_pretrained_ae_config(cfg, ckpt_cfg):
    """
    Check arguments are the same with pretrained model (old experiment)
    """
    #assert cfg['exp_name'] == ckpt_cfg['exp_name']

    # Data
    assert cfg['data'].dataset_name == ckpt_cfg['data'].dataset_name
    assert cfg['data'].dataset_path == ckpt_cfg['data'].dataset_path
    assert cfg['data'].sampling_rate == ckpt_cfg['data'].sampling_rate
    assert cfg['data'].window_in == ckpt_cfg['data'].window_in
    assert cfg['data'].window_out == ckpt_cfg['data'].window_out

    # Network
    assert cfg['network'].autoencoder == ckpt_cfg['network'].autoencoder
    assert cfg['network'].autoencoder_config == ckpt_cfg['network'].autoencoder_config
    #assert cfg['network'].loss_type == ckpt_cfg['network'].loss_type
    #assert cfg['network'].loss_ratio == ckpt_cfg['network'].loss_ratio

    # Training
    #print("Pass cfg.n_epochs")
    #print("Pass cfg.optimizer_dict")
    #assert cfg['training'].lr == ckpt_cfg['training'].lr
    #assert cfg['training'].lr_sched_dict['type'] == ckpt_cfg['training'].lr_sched_dict['type']
    #assert cfg['training'].path_batch_size == ckpt_cfg['training'].path_batch_size
    #assert cfg['training'].batch_size == ckpt_cfg['training'].batch_size
    #assert cfg['training'].seed == ckpt_cfg['training'].seed

    # Mode


# ============================================================================


# Import trainers
import torch
import torch.nn as nn

# Custom loss functions
from .recon_msd_mpd import LossReconMSDMPD
from .recon_msd_mpd_contrastive import LossReconMSDMPDContrastive

class LossNone(nn.Module):
    def __init__(self, losses, loss_weight, params):
        super(LossNone, self).__init__()

        self.loss_list = loss_list
        self.loss_weight = loss_weight
        
        self.loss_function_dict = {}



class SampleLoss(nn.Module):
    def __init__(self, loss_list, loss_weight, params=None):
        super(SampleLoss, self).__init__()

        self.loss_list = loss_list
        self.loss_weight = loss_weight
        
        self.loss_function_dict = {}


all_losses_dict = {
        'LossNone': LossNone,
        'LossReconMSDMPD': LossReconMSDMPD,
        'LossReconMSDMPDContrastive': LossReconMSDMPDContrastive,
        'SampleLoss': SampleLoss,
        }


def define_loss(loss_type, loss_list, loss_weight, params=None):
    # NOTE: Add new loss here

    return all_losses_dict[loss_type](loss_list, loss_weight, params)


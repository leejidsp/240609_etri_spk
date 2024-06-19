from .trainer_hifi_basic import TrainerHiFiBasic
from .trainer_hifi_autovc_raw import TrainerHiFiAutoVCRaw
from .trainer_hifi_autovc_mean import TrainerHiFiAutoVCMean

all_trainers_dict = {
        'TrainerHiFiBasic': TrainerHiFiBasic,
        'TrainerHiFiAutoVCRaw': TrainerHiFiAutoVCRaw,
        'TrainerHiFiAutoVCMean': TrainerHiFiAutoVCMean,
        }

#TODO: modify if needed
def define_trainer(trainer, data_dict, model_dict, opt_dict, loss_function, cfg):
    """
    Define trainer class according to model configuration
    """
    assert trainer in all_trainers_dict.keys(), NotImplementedError("Not implemented trainer")

    return all_trainers_dict[trainer](data_dict, model_dict, opt_dict, loss_function, cfg)
    



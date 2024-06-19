from .sample_model import SampleModel, SampleDiscriminator
from .m01_autovc_raw import M01AutoVCRaw
from .m02_same_vq import M02SameVQ
from .m03_multiple_vq import M03MultipleVQ
from .m04_adaptive_vq import M04AdaptiveVQ
from .m05_autovc_mean import M05AutoVCMean
from .hifigan_discriminator import MultiScaleDiscriminator, MultiPeriodDiscriminator

__all__ = [
            M01AutoVCRaw,
            M02SameVQ,
            M03MultipleVQ,
            M04AdaptiveVQ,
            M05AutoVCMean,

            SampleModel,
            SampleDiscriminator,

            MultiScaleDiscriminator,
            MultiPeriodDiscriminator,
            ]

all_model_dict ={
        'M01AutoVCRaw': M01AutoVCRaw,
        'M02SameVQ': M02SameVQ,
        'M03MultipleVQ': M03MultipleVQ,
        'M04AdaptiveVQ': M04AdaptiveVQ,
        'M05AutoVCMean': M05AutoVCMean,

        'SampleModel':	SampleModel,
        'SampleDiscriminator': SampleDiscriminator,
        
        'MultiScaleDiscriminator':  MultiScaleDiscriminator,
        'MultiPeriodDiscriminator': MultiPeriodDiscriminator,
        }


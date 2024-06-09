import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

#from .utils_model import init_weights, get_padding
#from utils_model import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class GeneratorHiFi(torch.nn.Module, ):
    def __init__(self, h, feature_dim):
        super(GeneratorHiFi, self).__init__()
        ### (0) Parameters
        self.h = h      #config(dict)
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        #print("num_kernels: ", self.num_kernels)
        #print("num_upsamples: ", self.num_upsamples)

        ### (1) Pre-convolution
        self.conv_pre = weight_norm(Conv1d(feature_dim, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 

        ### (2-1) Upsample blocks
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
        #print("len(ups): ", len(self.ups))
        #print(self.ups)

        ### (2-2) Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        #print("len(res): ", len(self.resblocks))    # =num_kernels per one upsample
        #print(self.resblocks)

        ### (3) Post-convolution
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        ### Initialize weights
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """ x: [B, num_mels, num_frames] """
        # (1)Pre
        x = self.conv_pre(x)        # [B, initial_channel, num_frames] 
        #print("Pre x: ", x.size()) 
        #print()

        # (2) Upsample & Residual
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)   # [B, initial_channel//(2**i), num_frames] 
            x = self.ups[i](x)  # [B, initial_channel//(2**(i+1)), num_frames*upsample_rate[i]]
            #print("{} 1: ".format(i), x.size())
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)    # 'num_kernels' resblocks per one upsample
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)   # Add
            x = xs / self.num_kernels   # Average
            #print("{} 2: ".format(i), x.size())
            #print(x.size())
        x = F.leaky_relu(x)
        #print("Res_final x: ", x.size()) 

        # (3) Post
        x = self.conv_post(x)
        #print("Post x: ", x.size()) 
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)



def init_weights(m, mean=0.0, std=0.01):
    #classname = m.__class__.__name__
    #if classname.find("Conv") != -1:
    #    m.weight.data.normal_(mean, std)
    #for ms in m.modules():
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std) 

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


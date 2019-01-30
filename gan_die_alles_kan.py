import torch

import torch.nn as nn
import torch.nn.functional as F

import utils
from cycle_gan import create_parser, print_opts
from data_loader import get_emoji_loader
from models import conv, ResnetBlock, deconv


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, noise_dim=2, init_zero_weights=False):
        super(CycleGenerator, self).__init__()
        self.conv_dim = conv_dim
        self.noise_dim = noise_dim

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(3, conv_dim, 4, padding=1, stride=2)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, padding=1, stride=2)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim*2)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(conv_dim * 2 + noise_dim, conv_dim, 4, padding=1, stride=2)
        self.deconv2 = deconv(conv_dim, 3, 4, padding=1, stride=2, batch_norm=False)

    def sample_noise(self, shape):
        return utils.to_var(torch.rand(*shape) * 2 - 1)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        # 8 x 8 x 64
        noise = self.sample_noise((out.shape[0], self.noise_dim, out.shape[2], out.shape[3]))

        # print(noise.shape)
        # print(out.shape)

        out = torch.cat([out, noise], dim=1)

        # print(out.shape)

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out



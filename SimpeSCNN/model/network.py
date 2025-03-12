import MinkowskiEngine as ME
import torch.nn as nn


class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
            ME.MinkowskiGlobalPooling(),
            ME.MinkowskiLinear(128, out_feat))

    def forward(self, x):
        return self.net(x)

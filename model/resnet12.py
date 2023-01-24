import torch
import torch.nn as nn
from functools import partial


class ResBlock(nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        super().__init__()
        self.convblock1 = ConvBlock(ni, nf, kss[0], 1)
        self.convblock2 = ConvBlock(nf, nf, kss[1], 1)
        self.convblock3 = ConvBlock(nf, nf, kss[2], 1)

        # expand channels for the sum if necessary
        self.shortcut = nn.BatchNorm1d(ni) if ni == nf else ConvBlock(ni, nf, 1, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.add(self.shortcut(res))
        x = self.act(x)
        return x


class ResNet(nn.Module):
    def __init__(self, c_in, c_out, data_loader, train_config):
        super().__init__()
        input_dim = data_loader.input_dimension
        sequence_len = data_loader.sequence_len
        output_dim = data_loader.output_dimension
        self.input_size = (sequence_len, input_dim)
        self.H_in = input_dim
        self.L = sequence_len
        self.H_out = output_dim

        # mid channels
        nf = 22

        kss = [7, 5, 3]
        self.resblock1 = ResBlock(c_in, nf, kss=kss)

        ## Only 1 resblock for now, will extend in future

        self.resblock2 = ResBlock(nf, nf, kss=kss)
        self.resblock3 = ResBlock(nf, nf, kss=kss)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nf, c_out) ## was nf * 2 for deep resnet

    def forward(self, x):
        x = self.resblock1(x)

        ## Only 1 resblock for now, will extend in future

        x = self.resblock2(x)
        x = self.resblock3(x)

        x = torch.squeeze(self.gap(x), dim=-1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class ConvBlock(nn.Module):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv1dSamePadding(nn.Module):
    "Conv1d with padding='same'"

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.conv1d_same = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = Pad1d

    def forward(self, x):
        self.padding = same_padding1d(x.shape[0], self.kernel_size, dilation=self.dilation) #stride=self.stride not used in padding calculation!
        return self.conv1d_same(self.pad(self.padding)(x))


class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)


def same_padding1d(seq_len, ks, stride=1, dilation=1):
    "Same padding formula as used in Tensorflow"
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2

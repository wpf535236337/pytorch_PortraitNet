#######################
# name: PortraitNet full model definition reproduced by Pytorch(v0.4.1)
# data: May 2019
# author:PengfeiWang(pfw813@gmail.com)
# paper: PortraitNet: Real-time Portrait Segmentation Network for Mobile Device
#######################

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from MobileNetV2 import MobileNetV2


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        #dw
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        #pw
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x =  F.relu(self.bn(self.conv(x)))
        x = self.pointwise(x)
        return x


class TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv1 = SeparableConv2d(in_channels,out_channels,3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = SeparableConv2d(out_channels,out_channels,3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, upsample_type ='deconv'):
        super(DecoderBlock, self).__init__()

        # as the paper 4.4.4. Speed analysis
        # we use bilinear interpolation based up-sampling instead of deconvolution in PortraitNet.
        assert upsample_type in ['deconv', 'bilinear']

        self.tansBlock = TransitionBlock(in_channels,out_channels)

        if upsample_type == 'deconv':
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, padding=1, stride=2,bias=False)
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):

        x = self.tansBlock(x)
        out = self.upsample(x)

        return out


class PortraitNet(nn.Module):
    def __init__(self, n_seg = 2,input_size=224, width_mult = 1.,n_class=None):
        '''
        :param n_seg:  class number of the segmentation
        :param input_size: the image size
        :param width_mult:  width multiplier parameter
        :param n_class:calss number of the classification pretrain model
        '''
        super(PortraitNet, self).__init__()

        self.backbone = MobileNetV2(input_size,width_mult,n_class)
        self.stage_output_channels = self.backbone.stage_output_channels[::-1]

        self.dblock1 = DecoderBlock(self.stage_output_channels[0], self.stage_output_channels[1])
        self.dblock2 = DecoderBlock(self.stage_output_channels[1] * 2, self.stage_output_channels[2])
        self.dblock3 = DecoderBlock(self.stage_output_channels[2] * 2, self.stage_output_channels[3])
        self.dblock4 = DecoderBlock(self.stage_output_channels[3] * 2, self.stage_output_channels[4])
        self.dblock5 = DecoderBlock(self.stage_output_channels[4] * 2, 3)

        self.mask_head = nn.Conv2d(3, n_seg, 1,bias=False)
        self.boundry_head = nn.Conv2d(3, 2, 1,bias=False)

        self._init_weights()


    def forward(self, x):

        # ====================== encoder part ================================================
        # according to  paper MobileNetV2: Inverted Residuals and Linear Bottlenecks
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x

        # ======================= decoder part====================================================
        up1 = self.dblock1(x5)

        up2 = self.dblock2(torch.cat([x4, up1], dim=1))

        up3 = self.dblock3(torch.cat([x3, up2], dim=1))

        up4 = self.dblock4(torch.cat([x2, up3], dim=1))

        up5 = self.dblock5(torch.cat([x1, up4], dim=1))

        mask = self.mask_head(up5)
        boundry = self.boundry_head(up5)

        return mask,boundry

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def getNetParams(net):

    pM = sum(p.numel() for p in net.parameters()) / float(1024 * 1024)

    return round(pM, 2)


def getFileSize(file_path):
    # file_path = unicode(file_path, 'utf8')
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


if __name__ == '__main__':

    input = Variable(torch.randn(1, 3, 224, 224))
    net = PortraitNet()
    out,_ = net(input)
    print(out.size())

    #
    print('total parameter:', getNetParams(net))
    model_name = ' PortraitNet.pth'
    torch.save(net.state_dict(), model_name)
    print('model size:', getFileSize(model_name), 'MB')
    os.remove(model_name)



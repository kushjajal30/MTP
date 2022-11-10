import torch
from torch import nn


def singlelayer(in_ch, out_ch, kernel, padding, stride, batchnorm):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, padding=padding, stride=stride)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.l1 = singlelayer(in_channel, out_channel, 3, 'same', 1, True)
        self.l2 = singlelayer(out_channel, out_channel, 3, 'same', 1, True)
        self.l3 = singlelayer(out_channel, out_channel, 3, 'same', 1, True)
        self.pool = nn.Conv2d(out_channel, out_channel, 2, padding='valid', stride=2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.pool(x)


class DeconvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DeconvBlock, self).__init__()
        self.l1 = singlelayer(in_channel, out_channel, 3, 'same', 1, True)
        self.l2 = singlelayer(out_channel, out_channel, 3, 'same', 1, True)
        self.l3 = singlelayer(out_channel, out_channel, 3, 'same', 1, True)
        self.unpool = nn.ConvTranspose2d(out_channel, out_channel, 2, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.unpool(x)


class GenUnet(nn.Module):

    def __init__(self,n):
        super(GenUnet, self).__init__()

        self.down_b1 = ConvBlock(n, 32)
        self.down_b2 = ConvBlock(32, 64)
        self.down_b3 = ConvBlock(64, 128)
        self.down_b4 = ConvBlock(128, 256)
        self.down_b5 = ConvBlock(256, 512)

        self.up_b5 = DeconvBlock(512, 256)
        self.up_b4 = DeconvBlock(512, 128)
        self.up_b3 = DeconvBlock(256, 64)
        self.up_b2 = DeconvBlock(128, 32)
        self.up_b1 = DeconvBlock(64, 32)

        self.final_decoder_1 = singlelayer(32, 1, 3, 'same', 1, True)
        self.final_decoder_2 = singlelayer(1, 1, 3, 'same', 1, True)
        self.final_decoder_3 = nn.Conv2d(1, 1, 3, padding='same')
        self.final_act = nn.Tanh()

    def forward(self, x):
        down_b1 = self.down_b1(x)
        down_b2 = self.down_b2(down_b1)
        down_b3 = self.down_b3(down_b2)
        down_b4 = self.down_b4(down_b3)
        down_b5 = self.down_b5(down_b4)

        up_b5 = self.up_b5(down_b5)
        up_b4 = self.up_b4(torch.cat([up_b5, down_b4], dim=1))
        up_b3 = self.up_b3(torch.cat([up_b4, down_b3], dim=1))
        up_b2 = self.up_b2(torch.cat([up_b3, down_b2], dim=1))
        up_b1 = self.up_b1(torch.cat([up_b2, down_b1], dim=1))

        out = self.final_decoder_1(up_b1)
        out = self.final_decoder_2(out)
        out = self.final_decoder_3(out)

        return self.final_act(out)

class Discriminator(nn.Module):

    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        self.l1 = singlelayer(input_channels, 64, 3, 'valid', 2, False)
        self.l2 = singlelayer(64, 128, 3, 'valid', 2, False)
        self.l3 = singlelayer(128, 256, 3, 'valid', 2, False)
        self.l4 = singlelayer(256, 256, 3, 'valid', 2, False)
        self.outlayer = nn.Conv2d(256, 1, 3, padding="valid", stride=2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.outlayer(x)
        
        return self.activation(x)

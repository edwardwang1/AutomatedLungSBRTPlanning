import numpy as np
import torch
from torch import nn

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight)

class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDownBlock, self).__init__()
        self.pipeline = nn.Sequential(
            # nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(ResidualBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout)
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        x = self.pipeline(x)
        x = nn.functional.pad(x, (1, 0, 1, 0, 1, 0))
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class Attention_block(nn.Module):
    #Adapted from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, verbose=False):
        super(Generator, self).__init__()
        self.verbose = verbose

        num_features = [16, 32, 64, 64]

        self.first_layer = UNetDownBlock(in_channels, num_features[0])

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.bottlenecks = nn.ModuleList()
        for i in range(4):
            self.bottlenecks.append(ResidualBlock(num_features[-1] * 2, num_features[-1], dropout=0.2))

        self.ups = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(UNetUpBlock(num_features[-1] * 2, num_features[-1]))
            else:
                self.ups.append(UNetUpBlock(num_features[-i - 2] * 4, num_features[-i - 2]))

        self.last_layer = nn.Sequential(
            nn.ConvTranspose3d(num_features[0] * 2, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, cond):
        x = torch.cat((x, cond), dim=1)
        x = self.first_layer(x)
        skip_connections = []
        if self.verbose:
            print("after first layer", x.shape)
        for i, d in enumerate(self.downs):
            skip_connections.append(x)
            x = d(x)
            if self.verbose:
                print("down" + str(i), x.shape)

        skip_connections = skip_connections[::-1]

        #         for s in skip_connections:
        #             print("skip", s.shape)

        # Middle part
        for bottleneck in self.bottlenecks:
            x_prev = x
            x = bottleneck(torch.cat((x, x_prev), dim=1))
            if self.verbose:
                print("bottlneck", x.shape)

        for i in range(len(self.ups)):
            if i == 0:
                #   print(x.shape, x_prev.shape)
                u = self.ups[i]
                concat = torch.cat((x, x_prev), dim=1)
                x = u(concat)
                if self.verbose:
                    print("up" + str(i), x.shape)
            else:
                # print(x.shape, skip_connections[i-1].shape)
                u = self.ups[i]
                if x.shape != skip_connections[i - 1].shape:
                    difference = np.array(skip_connections[i - 1].shape) - np.array(x.shape)
                    #        print(difference)
                    x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
                    #         print("padded", x.shape, skip_connections[i-1].shape)
                concat = torch.cat((x, skip_connections[i - 1]), dim=1)
                #      print("--", concat.shape)
                x = u(concat)
                if self.verbose:
                    print("up" + str(i), x.shape)

        # print(x.shape)

        x = self.last_layer(torch.cat((x, skip_connections[-1]), dim=1))

        return x

class AttentionGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        #adapated from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
        super(AttentionGenerator, self).__init__()

        num_features = [16, 32, 64, 64]

        self.first_layer = UNetDownBlock(in_channels, num_features[0])

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.bottlenecks = nn.ModuleList()
        for i in range(4):
            self.bottlenecks.append(ResidualBlock(num_features[-1] * 2, num_features[-1], dropout=0.2))

        self.ups = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(UNetUpBlock(num_features[-1] * 2, num_features[-1]))
            else:
                self.ups.append(UNetUpBlock(num_features[-i - 2] * 4, num_features[-i - 2]))

        self.attentions = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.attentions.append(Attention_block(F_g=num_features[-1],F_l=num_features[-1], F_int=num_features[-i - 2] * 2))
            else:
                self.attentions.append(Attention_block(F_g=num_features[-i - 2] * 2, F_l=num_features[-i - 2] * 2, F_int=num_features[-i - 2]))


        self.last_layer = nn.Sequential(
            nn.ConvTranspose3d(num_features[0] * 2, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.first_layer(x)
        skip_connections = []
        for d in self.downs:
            skip_connections.append(x)
            x = d(x)

        skip_connections = skip_connections[::-1]

        #         for s in skip_connections:
        #             print("skip", s.shape)

        # Middle part
        for bottleneck in self.bottlenecks:
            x_prev = x
            x = bottleneck(torch.cat((x, x_prev), dim=1))

        for i in range(len(self.ups)):
            if i == 0:
                u = self.ups[i]
                attention_out = self.attentions[i](x, x_prev)
                concat = torch.cat((x, attention_out), dim=1)
                x = u(concat)
            else:
                # print(x.shape, skip_connections[i-1].shape)
                u = self.ups[i]
                if x.shape != skip_connections[i - 1].shape:
                    difference = np.array(skip_connections[i - 1].shape) - np.array(x.shape)
                    #        print(difference)
                    x = nn.functional.pad(x, (difference[3], 0, difference[4], 0, difference[2], 0))
                    #         print("padded", x.shape, skip_connections[i-1].shape)

                attention_out = self.attentions[i](x, skip_connections[i - 1])
                concat = torch.cat((x, attention_out), dim=1)
                #      print("--", concat.shape)
                x = u(concat)

        # print(x.shape)

        x = self.last_layer(torch.cat((x, skip_connections[-1]), dim=1))

        return x

class Discriminator(nn.Module):
    def __init__(self, in_features=3, last_conv_kernalsize=4, verbose=False):
        super(Discriminator, self).__init__()

        self.verbose = verbose

        num_features = [in_features, 16, 32, 64, 128]

        self.downs = nn.ModuleList()
        self.num_layers = len(num_features) - 1
        for i in range(self.num_layers):
            self.downs.append(UNetDownBlock(num_features[i], num_features[i + 1]))

        self.last_layer = nn.Sequential(
            nn.Conv3d(num_features[-1], 1, last_conv_kernalsize, 1, 1),
            #nn.Sigmoid()
        )

    def forward(self, x, alt_cond):
        x = torch.cat((x, alt_cond), dim=1)
        for d in self.downs:
            x = d(x)

        if self.verbose:
            print("before last layer", x.shape)
        x = self.last_layer(x)
        if self.verbose:
            print("after last layer", x.shape)


        return x

class DenseConvolve(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseConvolve, self).__init__()
        self.pipeline = nn.Sequential(nn.Conv3d(in_channels, out_channels - in_channels, 3, 1, 1),
                    nn.ReLU(inplace=True)
                                      )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        x2 = self.pipeline(x)
        return torch.cat((x, x2), dim=1)

class DenseDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseDownsample, self).__init__()
        self.pipeline = nn.Sequential(nn.Conv3d(in_channels, out_channels - in_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(2)
        self.pipeline.apply(init_weights)

    def forward(self, x):
        x2 = self.pipeline(x)
        x = self.pool(x)

        return torch.cat((x, x2), dim=1)

class HDUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(HDUNetUpBlock, self).__init__()
        self.pipeline = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pipeline.apply(init_weights)

    def forward(self, x):
        return self.pipeline(x)

class DenseConvolvePaired(nn.Module):
    def __init__(self, in_channels, channel_increment=16):
        super(DenseConvolvePaired, self).__init__()
        self.DenseConvolve1 = DenseConvolve(in_channels, in_channels + channel_increment)
        self.DenseConvolve2 = DenseConvolve(in_channels + channel_increment, in_channels + 2 * channel_increment)

    def forward(self, x):
        x = self.DenseConvolve1(x)
        x = self.DenseConvolve2(x)
        return x

class HDUnet(nn.Module):
    def __init__(self, in_channels, out_channels=1, verbose=False):
        super(HDUnet, self).__init__()
        self.verbose = verbose

        #Increment num features by 16 for every convolution operation
        self.CHANNEL_INCREMENT = 16

        self.DenseConvolvePaired1 = DenseConvolvePaired(in_channels, self.CHANNEL_INCREMENT)
        self.DenseDownsample1 = DenseDownsample(in_channels + 2 * self.CHANNEL_INCREMENT, in_channels + 3 * self.CHANNEL_INCREMENT)
        self.DenseConvolvePaired2 = DenseConvolvePaired(in_channels + 3 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseDownsample2 = DenseDownsample(in_channels + 5 * self.CHANNEL_INCREMENT, in_channels + 6 * self.CHANNEL_INCREMENT)
        self.DenseConvolvePaired3 = DenseConvolvePaired(in_channels + 6 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseDownsample3 = DenseDownsample(in_channels + 8 * self.CHANNEL_INCREMENT, in_channels + 9 * self.CHANNEL_INCREMENT)
        self.DenseConvolvePaired4 = DenseConvolvePaired(in_channels + 9 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.Downsample4 = DenseDownsample(in_channels + 11 * self.CHANNEL_INCREMENT, in_channels + 12 * self.CHANNEL_INCREMENT)
        self.DenseConvolvePaired5 = DenseConvolvePaired(in_channels + 12 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseConvolvePaired6 = DenseConvolvePaired(in_channels + 14 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseUpsample1 = HDUNetUpBlock(in_channels + 16 * self.CHANNEL_INCREMENT, in_channels + 15 * self.CHANNEL_INCREMENT - (in_channels + 11 * self.CHANNEL_INCREMENT))
        self.DenseConvolvePaired7 = DenseConvolvePaired(in_channels + 15 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseUpsample2 = HDUNetUpBlock(in_channels + 17 * self.CHANNEL_INCREMENT, in_channels + 12 * self.CHANNEL_INCREMENT - (in_channels + 8 * self.CHANNEL_INCREMENT))
        self.DenseConvolvePaired8 = DenseConvolvePaired(in_channels + 12 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseUpsample3 = HDUNetUpBlock(in_channels + 14 * self.CHANNEL_INCREMENT, in_channels + 9 * self.CHANNEL_INCREMENT - (in_channels + 5 * self.CHANNEL_INCREMENT))
        self.DenseConvolvePaired9 = DenseConvolvePaired(in_channels + 9 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.DenseUpsample4 = HDUNetUpBlock(in_channels + 11 * self.CHANNEL_INCREMENT, in_channels + 6 * self.CHANNEL_INCREMENT - (in_channels + 2 * self.CHANNEL_INCREMENT))
        self.DenseConvolvePaired10 = DenseConvolvePaired(in_channels + 6 * self.CHANNEL_INCREMENT, self.CHANNEL_INCREMENT)
        self.FinalConv = nn.Conv3d(in_channels + 8 * self.CHANNEL_INCREMENT, out_channels, 3, 1, 1)


    def forward(self, x):
        skip_connections = []
        x = self.DenseConvolvePaired1(x)
        if self.verbose:
            print(x.shape)
        skip_connections.append(x)
        x = self.DenseDownsample1(x)
        x = self.DenseConvolvePaired2(x)
        if self.verbose:
            print(x.shape)
        skip_connections.append(x)
        x = self.DenseDownsample2(x)
        x = self.DenseConvolvePaired3(x)
        if self.verbose:
            print(x.shape)
        skip_connections.append(x)
        x = self.DenseDownsample3(x)
        x = self.DenseConvolvePaired4(x)
        if self.verbose:
            print(x.shape)
        skip_connections.append(x)
        x = self.Downsample4(x)
        x = self.DenseConvolvePaired5(x)
        if self.verbose:
            print(x.shape)
        x = self.DenseConvolvePaired6(x)
        if self.verbose:
            print(x.shape)
        x = self.DenseUpsample1(x)
        x = torch.cat((x, skip_connections.pop()), dim=1)
        x = self.DenseConvolvePaired7(x)
        if self.verbose:
            print(x.shape)
        x = self.DenseUpsample2(x)
        x = torch.cat((x, skip_connections.pop()), dim=1)
        x = self.DenseConvolvePaired8(x)
        if self.verbose:
            print(x.shape)
        x = self.DenseUpsample3(x)
        x = torch.cat((x, skip_connections.pop()), dim=1)
        x = self.DenseConvolvePaired9(x)
        if self.verbose:
            print(x.shape)
        x = self.DenseUpsample4(x)
        x = torch.cat((x, skip_connections.pop()), dim=1)
        x = self.DenseConvolvePaired10(x)
        if self.verbose:
            print(x.shape)
        x = self.FinalConv(x)

        return x


if __name__ == '__main__':

    model = HDUnet(10, 1, verbose=True)
    x = torch.randn((2, 10, 84, 128, 128))

    model = model.cuda()
    x = x.cuda()

    output = model(x)
    print(output.shape)
    #print(output)





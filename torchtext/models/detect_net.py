import torch.nn as nn
import torch
import torch.nn.functional as F
try:
    from vgg import *
    from resnet import *
    from senet import *
except:
    from .vgg import *
    from .resnet import *
    from .senet import *


class MergeUpsample(nn.Module):

    def __init__(self, in_channels, out_channels, conv3x3, p_dropout=0.1):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        if conv3x3:
            self.conv3x3or5x5 = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv3x3or5x5 = nn.Conv2d(
                in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p_dropout, inplace=False)
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3x3or5x5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.deconv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, multiply, p_dropout=0.1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p_dropout, inplace=False)
        self.multiply = multiply
        if multiply > 1:
            ratio = multiply/2
            self.deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=int(4*ratio), stride=int(2*ratio), padding=int(1*ratio))

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.multiply > 1:
            x = self.deconv(x)
        return x


resnet = {'resnet34': resnet34, 'resnet50': resnet50,
          'resnet101': resnet101, 'resnet152': resnet152}
vgg = {'vgg16': vgg16}

se_net = {'se_resnext50_32x4d': se_resnext50_32x4d}


class DetectNet(nn.Module):

    def __init__(self, backbone='vgg16', output_channel=2):
        super().__init__()

        self.backbone_name = backbone
        self.output_channel = output_channel

        if backbone[0:3] == 'vgg':
            self.backbone = vgg[backbone]()
            self._make_upsample(expansion=1)
        elif backbone[0:6] == 'resnet':
            self.backbone = resnet[backbone]()
            expansion = 1 if backbone in ['resnet18', 'resnet34'] else 4
            self._make_upsample(expansion=expansion)
        elif backbone == 'se_resnext50_32x4d':
            self.backbone = se_net[backbone]()
            self._make_upsample(expansion=4)

    def _make_upsample(self, expansion=1):
        if self.backbone_name == 'vgg16':
            self.deconv5 = nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=2, padding=1)
            self.merge4 = MergeUpsample(512 + 512, 256, True)
            self.merge3 = MergeUpsample(256 + 256, 128, True)
            self.merge2 = MergeUpsample(128 + 128, 64, True)
            self.merge1 = MergeUpsample(64 + 64, self.output_channel, True)
        else:
            # self.deconv5 = nn.ConvTranspose2d(
            #     512*expansion, int(512*expansion/2), kernel_size=4, stride=2, padding=1)
            self.deconv5 = Upsample(512*expansion, int(512*expansion/2), 2)
            self.merge4 = MergeUpsample(
                512*expansion/2+256*expansion, 256*expansion/2, True)
            self.merge3 = MergeUpsample(
                256*expansion/2 + 128*expansion, 128*expansion/2, True)
            self.merge2 = MergeUpsample(
                128*expansion/2 + 64*expansion, 64*expansion/2, False)
            self.merge1 = MergeUpsample(64*expansion/2 + 64, 64, False)
            self.predict = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, self.output_channel,
                          kernel_size=1, stride=1, padding=0)
            )
            # self.merge3x2 = Upsample(128*expansion, 128*expansion, 2)
            # self.up2x2 = Upsample(128*expansion, 128*expansion, 2)
            # self.up2 = nn.Sequential(
            #     nn.Conv2d(128*expansion, 128*expansion,
            #               kernel_size=1, stride=1, padding=0),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(128*expansion, 128*expansion,
            #               kernel_size=1, stride=1, padding=0),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(128*expansion, self.output_channel,
            #               kernel_size=1, stride=1, padding=0)
            # )
            # ratio = 4/2
            # self.up1x2 = Upsample(128*expansion, self.output_channel, 2)
            # self.up1x4 = nn.ConvTranspose2d(
            #     self.output_channel, self.output_channel, kernel_size=int(4*ratio), stride=int(2*ratio), padding=int(1*ratio))

    def forward(self, x):
        # print('xxxxxxxxxxxx size',x.size())
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        # up5 = self.relu(up5)

        # print('c4 up5',C4.size(),up5.size())
        up4 = self.merge4(C4, up5)
        # up4 = self.relu(up4)

        # print('c3 up4',C3.size(),up4.size())
        up3 = self.merge3(C3, up4)
        # up3 = self.relu(up3)
        # up3 = self.merge3x2(up3)

        # print('c2 up3',C2.size(),up3.size())
        up2 = self.merge2(C2, up3)
        # up2 = self.relu(up2)
        # up2 = self.up2x2(up3)

        # print('c1 up2',C1.size(),up2.size())
        up1 = self.merge1(C1, up2)
        # up1 = self.relu(up1)
        # up1 = self.up1x2(up2)
        predict = self.predict(up1)
        return predict


if __name__ == '__main__':
    import torch
    input = torch.randn((1, 3, 512, 512))
    net = DetectNet(backbone='resnet50').cuda()
    print(net(input.cuda()).size())
    import torchsummary
    # with torch.no_grad():
    print(torchsummary.summary(net, (3, 512, 512), batch_size=1))
    exit()

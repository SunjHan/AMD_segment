import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation=1):
    return nn.Conv2d(in_planes, out_planes,kernel_size = 3, stride = stride, padding = dilation, 
            groups = groups, bias = False, dilation = dilation)

def conv1x1(in_planes, out_planes, stride=1, group=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1  #expansion=last_block_channel/first_block_channel
    __constant__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups!=1 or base_width!=64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) #inplace选择是否进行覆盖运算
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * 1, stride),
                norm_layer(planes * 1),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print("out shape: ",out.shape)
        # print("identity shape: ",identity.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class res_Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_Unet, self).__init__()

        self.conv1 = BasicBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = BasicBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = BasicBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = BasicBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = BasicBlock(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = BasicBlock(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = BasicBlock(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = BasicBlock(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = BasicBlock(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)

        return out


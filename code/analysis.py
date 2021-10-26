# -------
# 定义各类网络
# -------
from torchvision.transforms import ToPILImage
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.utils.data as data
from pathlib import Path
import math
import time
import random
import time
import xlwt
import cv2
# from pytorch.org
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# from pytorch.org
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# from pytorch.org,resnet基本块实现
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# from pytorch.org,resnet基本块实现
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# airnext基本块实现
class AIRXBottleneck(nn.Module):
    """
    AIRXBottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, ratio=2, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width
            cardinality: num of convolution groups
            stride: conv stride. Replaces pooling layer
            ratio: dimensionality-compression ratio.
        """
        super(AIRXBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64.0)))
        C = cardinality
        self.stride = stride
        self.planes = planes

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if self.stride == 1 and self.planes < 512:  # for C2, C3, C4 stages
            self.conv_att1 = nn.Conv2d(inplanes, D * C // ratio, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_att1 = nn.BatchNorm2d(D * C // ratio)
            self.subsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.conv_att2 = nn.Conv2d(D*C // ratio, D*C // ratio, kernel_size=3, stride=2, padding=1, groups=C//2, bias=False)
            # self.bn_att2 = nn.BatchNorm2d(D*C // ratio)
            self.conv_att3 = nn.Conv2d(D * C // ratio, D * C // ratio, kernel_size=3, stride=1,
                                       padding=1, groups=C // ratio, bias=False)
            self.bn_att3 = nn.BatchNorm2d(D * C // ratio)
            self.conv_att4 = nn.Conv2d(D * C // ratio, D * C, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_att4 = nn.BatchNorm2d(D * C)
            self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.stride == 1 and self.planes < 512:
            att = self.conv_att1(x)
            att = self.bn_att1(att)
            att = self.relu(att)
            # att = self.conv_att2(att)
            # att = self.bn_att2(att)
            # att = self.relu(att)
            att = self.subsample(att)
            att = self.conv_att3(att)
            att = self.bn_att3(att)
            att = self.relu(att)
            att = F.interpolate(att, size=out.size()[2:], mode='bilinear')
            att = self.conv_att4(att)
            att = self.bn_att4(att)
            att = self.sigmoid(att)
            out = out * att

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 构建seresneext的sebottleneck的最后一层
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # batch*channels->batch*channels//reduction
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # batch*channels//reduction->batch*channels
            nn.Sigmoid()  # batch*channels
        )

    def forward(self, x):
        """
        inputsize和outputsize相同，element-wise乘法
        x:batch*channels*height*width
        """

        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)  # batch*channels
        y = self.fc(y).view(batch_size, channels, 1, 1)  # batch*channels->batch*channels*1*1
        return x * y.expand_as(x)  # 将y整理为与x一样的size然后做element_wise乘法，得到attention结果


# 实现seresnet50的基本块
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)  # 直接覆盖，减少显存消耗
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# channel_attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)  # 原位提换，减少显存消耗
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# spatial_attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 引入attention的resnet实现，以torch的resnet源码加以改造实现
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3474, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(self.inplanes)
        self.sa1 = SpatialAttention()

        # 在网络的feature_map层加入注意力机制
        self.ca2 = ChannelAttention(128 * block.expansion)
        self.sa2 = SpatialAttention()
        self.fc2 = nn.Linear(128 * block.expansion, num_classes)  # 用于feature_map层的线性变换

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ca(x) * x  # channel_attention
        x = self.sa(x) * x  # spatial_attention

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        """
        feature_map=self.ca2(x)*x
        feature_map=self.sa2(feature_map)*feature_map
        feature_map=self.avgpool(feature_map)
        feature_map=feature_map.reshape(feature_map.size(0),-1)#batch*channels
        feature_map=self.fc2(feature_map)#batch*num_classes
        """

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ca1(x) * x  # channel_attention
        x = self.sa1(x) * x  # spatial_attention

        x = self.avgpool(x)  # batch*channels*1*1
        x = x.reshape(x.size(0), -1)  # batch*channels
        x = self.fc(x)  # batch*num_classes

        return x

    def spe_attention(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ca(x) * x  # channel_attention
        x = self.sa(x) * x  # spatial_attention

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        """
        feature_map=self.ca2(x)*x
        feature_map=self.sa2(feature_map)*feature_map
        feature_map=self.avgpool(feature_map)
        feature_map=feature_map.reshape(feature_map.size(0),-1)#batch*channels
        feature_map=self.fc2(feature_map)#batch*num_classes
        """

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ca1(x) * x  # channel_attention
        x = self.sa1(x) * x  # spatial_attention
        return x
    # airnext的实现


class AirNext(nn.Module):
    def __init__(self, baseWidth=4, cardinality=32, head7x7=True, ratio=2, layers=(3, 4, 23, 3), num_classes=3474):
        """ Constructor
        Args:
            baseWidth: baseWidth for AIRX.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(AirNext, self).__init__()
        block = AIRXBottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, ratio)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, ratio)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, ratio)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ratio=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, ratio, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, 1, ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# airnex
def airnext50(num_classes=3474):
    model = AirNext(baseWidth=4, cardinality=32, head7x7=False, layers=(3, 4, 6, 3),
                    num_classes=num_classes)  # 同resnext一样设置了width和cardinality
    return model


# se-resnext50
def se_resnext50(num_classes=3474):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=32,
                   width_per_group=4)  # 设置groups构建为resnext
    return model


def resnext50(num_classes=3474):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)
    return model


def se_resnet50(num_classes=3474):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)  # 不设置groups和width_per_group
    # model = vision.models.resnet50(num_classes = num_classes)
    return model


# se_resnext101
def se_resnext101(num_classes=3474):
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, groups=32, width_per_group=8)
    return model


# resnext101
def resnext101(num_classes=3474):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, groups=32, width_per_group=8)
    return model


# resnet101
def se_resnet101(num_classes=3474):
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model

def resnet101(num_classes=3474):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model
def resnet50(num_classes=3474):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)#不设置groups和width_per_group
    #model = vision.models.resnet50(num_classes = num_classes)
    return model

class RandomErasing(object):
    '''
    probability:执行擦除操作的概率（体现随机性）
    sl: 擦除面积的下限
    sh: 擦除面积的上限
    r1: 擦除区域的长宽比界限，取区间（rl,1/rl）
    mean: erasing value
    '''

    def __init__(self, probability=0.3, sl=0.02, sh=0.3, r1=0.3, mean=[0.485, 0.456, 0.406]):
        # 这里mean的参数设置可调整
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # 按一定的概率决定是否执行Erasing操作
        if np.random.uniform(0, 1) > self.probability:
            return img

        area = img.size()[1] * img.size()[2]
        for attempt in range(100):  # 这里的100次可调整
            target_area = np.random.uniform(self.sl, self.sh) * area  # 目标擦除区域面积
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)  # 目标擦除区域宽高比

            # 计算目标擦除区域的宽和高
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                # 随机选取擦除区域：通过随机选出擦除区域左上角点的坐标得到
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:  # RGB图像用这个
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:  # 非RGB
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomResizedCropV2(transforms.RandomResizedCrop):

    @staticmethod
    def get_params(img, scale, ratio):
        # ...

        # fallback
        w = min(img.size[0], img.size[1])
        i = random.randint(0, img.size[1] - w)
        j = random.randint(0, img.size[0] - w)

        return i, j, w, w


# 数据预处理
trans_train = transforms.Compose([
    # HorizontalFlip(p=0.5),
    # OneOf([
    #        RandomBrightness(0.1, p=1),
    #        RandomContrast(0.1, p=1),
    # ], p=0.3),
    # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
    # IAAAdditiveGaussianNoise(p=0.3),
    RandomResizedCropV2((288, 288), scale=(0.7, 1.0), ratio=(4 / 5, 5 / 4)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]),
    RandomErasing()

])
trans_valid = transforms.Compose([
    RandomResizedCropV2((288, 288), scale=(0.7, 1.0), ratio=(4 / 5, 5 / 4)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])


# 定义第一层dataset类
class MyDataset_stack1(Dataset):  # MyDataset继承了Dataset类，该类并不保存图片，只是在读取时选择对应图片路径并返回图片
    def __init__(self, df_data, mode, data_dir='./', transform=None):
        """
        args:
        id_data：panda读出的csv数据,已经转化为list of (id,attribute_ids(为string类型),list of (attribute_ids))
        mode:'train'或者'test'
        data_dir:表明图片文件夹所在路径
        transform:对图片的transform操作
        """
        super().__init__()
        self.df = df_data
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):  # 获取元素时候再读入图片
        if (self.mode == 'train'):
            img_name = self.df[index][0]
        else:
            img_name = self.df[index][0]
        img_path = os.path.join(self.data_dir, img_name + '.png')  # 有关于系统路径问题
        image = Image.open(img_path).convert('RGB')  # 转化为RGB图像
        image = self.transform(image)  # 用对应的transform预处理
        if (self.mode == 'train'):
            label = self.df[index][2]
            label_tensor = np.zeros((1, 3474))
            for i in label:
                label_tensor[0, int(i)] = 1
            label_tensor = label_tensor.flatten()  # 变为一维
            label_tensor = torch.from_numpy(label_tensor).float()
            return image, label_tensor
        else:
            return image


# 定义第二层dataset类
class MyDataset_stack2(Dataset):
    def __init__(self, mode, data_dir, length, id_data):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.length = length
        self.id_data = id_data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.mode == 'valid': index = 5 * 23680 + index  # 最后一折数据
        vector = pd.read_csv(self.data_dir, index_col=index)
        vector = torch.Tensor(vector).reshape(3, 18, 193)  # 重新整理规模
        label = self.id_data[index][2]  # 取得labels数据
        label_tensor = np.zeros((1, 3474))
        for i in label:
            label_tensor[0, int(i)] = 1
        label_tensor = label_tensor.flatten()  # 变为一维
        label_tensor = torch.from_numpy(label_tensor).float()
        return vector, label_tensor


# 定义分类器与其各功能
class Stack1(nn.Module):
    def __init__(self, num_classes, lr, device, num_epoches, batch_size, threshold, model_name, data_dir, valid_fold,
                 weight_dacay):
        super(Stack1, self).__init__()

        # 超参数设定
        self.num_classes = num_classes
        self.lr = lr
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_name = model_name
        self.data_dir = data_dir
        self.device = device
        self.bound = [0, 100, 781, 786, 2706, 3474]
        self.valid_fold = valid_fold
        self.weight_dacay = weight_dacay

        # 模型选择
        if model_name == 'se_resnet50':
            self.model = se_resnet50(num_classes=3474).to(self.device)
        if model_name == 'se_resnext50':
            self.model = se_resnext50(num_classes=3474).to(self.device)
        if model_name == 'resnext50':
            self.model = resnext50(num_classes=3474).to(self.device)
        if model_name == 'airnext50':
            self.model = airnext50(num_classes=3474).to(self.device)
        if model_name == 'se_resnet101':
            self.model = se_resnet101(num_classes=3474).to(self.device)
        if model_name == 'se_resnext101':
            self.model = se_resnext101(num_classes=3474).to(self.device)
        if model_name == 'resnext101':
            self.model = resnext101(num_classes=3474).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')  # 对最后一层(线形层之后的结果)每个神经元做逻辑斯蒂回归
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_dacay)  # 第一个阶梯的lr

        # 数据导入
        self.dataset_train, self.dataset_valid, self.dataset_test = self.get_dataset()
        self.parameter_path = data_dir + model_name + str(self.valid_fold) + '.pth'  # 生成参数的保存路径

    # 用于导入数据并返回对应的dataset对象,传入数据只有路径
    # 分折训练32*740*6=142080（总数据量为142119）
    def get_id_data(self):

        labels = pd.read_csv(self.data_dir + "labels.csv")
        train_id_data = pd.read_csv(self.data_dir + "train.csv")
        test_id_data = pd.read_csv(self.data_dir + "sample_submission.csv")

        y = train_id_data.attribute_ids.map(lambda x: x.split()).values  # 将attribute每个值进行划分，因为一个值包含多个标签
        train_id_data['y'] = y  #:增加纵向列为label(int)的lsit
        test_id_data = test_id_data['id']

        unfold_train_id_data = train_id_data.values  # 用values可以取到ndarray of (id,attribute_ids(为string类型),list of (attribute_ids))
        test_id_data = test_id_data.values  # 用values可以取到ndarray of id

        return unfold_train_id_data, test_id_data

    # 得到dataset类
    def get_dataset(self):

        train_path = self.data_dir + 'train/'
        test_path = self.data_dir + 'test/'
        unfold_train_id_data, test_id_data = self.get_id_data()  # 提取id_data为ndarray
        # 将多余的数据补充到最后一折中
        if self.valid_fold == 5:
            valid_id_data = unfold_train_id_data[0:500]
            #train_id_data = unfold_train_id_data[0: 23680 * 5]
        else:
            valid_id_data = unfold_train_id_data[0:500]
            #train_id_data = np.vstack(
            #    (unfold_train_id_data[0:23680 * self.valid_fold], unfold_train_id_data[23680 * (self.valid_fold + 1):]))
        # train_id_data,valid_id_data = train_test_split(train_id_data,test_size=0.1)
        # 传入的df_data已经是values
        dataset_train = MyDataset_stack1(df_data=unfold_train_id_data, mode='test', data_dir=train_path,
                                         transform=trans_valid)
        dataset_valid = MyDataset_stack1(df_data=valid_id_data, mode='train', data_dir=train_path,
                                         transform=trans_valid)
        dataset_test = MyDataset_stack1(df_data=test_id_data, mode='test', data_dir=test_path, transform=trans_valid)
        print('导入数据结束')
        return dataset_train, dataset_valid, dataset_test

    # 参数训练
    def train(self, pre_trained=False, min_loss=np.inf):
        loader_train = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        loader_valid = DataLoader(dataset=self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=8)
        min_val_loss = min_loss  # 记录最小loss
        sym, times = 0, 0  # 用来控制学习率的衰减,记录不减loss数和衰减次数
        if pre_trained:
            self.model.load_state_dict(torch.load('../input/ser501x/se_resnet501.pth'))
            print('pre_trained, loss:', min_loss)
        bestmodel = self.model.state_dict()  # 用来记录模型参数
        start_time = time.time()
        print('start training valid_fold', self.valid_fold)

        for epoch in range(self.num_epoches):
            start = time.time()
            avg_loss = 0
            # 训练
            for iteration, (images, labels) in enumerate(loader_train):
                images = images.to(self.device)  # shape:batch*in_channels*height*width
                labels = labels.to(self.device)  # shape:batch*num_classes
                self.optimizer.zero_grad()
                outputs = self.model.forward(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # if iteration%200==0:print('epoch', epoch,' iteration',iteration,'finished ,taking',(time.time()-start)/60,'min')

            # 在验证集上计算loss和f2
            current_F2 = self.eval(pre_trained=False)  # 计算f2，已在内部断开梯度累计
            with torch.no_grad():
                for i, (images, labels) in enumerate(loader_valid):
                    images = images.to(self.device)  # shape:batch*in_channels*height*width
                    labels = labels.to(self.device)  # shape:batch*num_classes
                    outputs = self.model.forward(images)  # shape:batch*num_calsses
                    loss = self.criterion(outputs, labels)
                    avg_loss += loss.item() / len(loader_valid)

            # 通过patience实现衰减机制
            if min_val_loss > avg_loss:
                bestmodel = self.model.state_dict()  # 以loss为标准保存最好模型
                min_val_loss = avg_loss
                sym = 0
            else:  # 不减
                sym += 1
                if (sym >= 2 and times < 3):
                    newlr = 0
                    sym = 0
                    times += 1
                    for p in self.optimizer.param_groups:
                        p['lr'] *= 1 / 7
                        newlr = p['lr']

                    print('learning rate decays to', newlr)

            # 输出
            print('epoch:[{}],current_loss:[{}]'.format(epoch + 1, avg_loss))
            print('耗时', (time.time() - start) / 60, 'min\n')

        torch.save(bestmodel, './se_resnet501.pth')
        print('训练结束，总耗时：', (time.time() - start_time) / 60)

    # 用最好模型计算在验证集上的F2（cv）,batch_size为默认batch-size而不是128否则容易炸显存
    def eval(self, pre_trained=False):
        if pre_trained:  # 如果pretrained，则load
            self.model.load_state_dict(torch.load(self.parameter_path))  # 加载模型参数到预定路径中
        TP, FN, FP, TN = 0, 0, 0, 0
        loader_valid = DataLoader(dataset=self.dataset_valid, batch_size=self.batch_size, shuffle=False,
                                  num_workers=8)  # 关于droplast，若为true则舍掉最后个不完整batch
        remain_num = (len(self.dataset_valid) - 1) % self.batch_size + 1  # 剩余部分,减一取模再加一，考虑整除情况
        ones = torch.ones(self.batch_size, self.num_classes).to(self.device)
        zeros = torch.zeros(self.batch_size, self.num_classes).to(self.device)
        ones_remain = torch.ones(remain_num, self.num_classes).to(self.device)  # 最后一部分剩余量的掩码比较
        zeros_remain = torch.zeros(remain_num, self.num_classes).to(self.device)
        threshold_list = []  # 用于构造界的tensor
        for i in range(5):
            threshold_list += [self.threshold[i]] * (self.bound[i + 1] - self.bound[i])
        threshold_tensor = torch.Tensor([threshold_list] * self.batch_size).to(self.device)

        with torch.no_grad():  # 避免梯度积累
            for iteration, (images, labels) in enumerate(loader_valid):
                if iteration == len(loader_valid) - 1:
                    ones, zeros = ones_remain, zeros_remain
                    threshold_tensor = torch.Tensor([threshold_list] * remain_num).to(self.device)
                images = images.to(self.device)  # 放在gpu上,此时的size是batch*5*channels*height*width
                labels = labels.to(self.device)  # 放在gpu上
                outputs = self.model.forward(images)  # shape:batch_size*num_classes，此时只是得到线性映射结果
                outputs = torch.sigmoid(outputs)  # sigmoid非线性激活

                outputs = outputs > threshold_tensor  # 二分为0-1矩阵
                TP += ((outputs == ones) & (labels == ones)).sum().item()
                FN += ((outputs == zeros) & (labels == ones)).sum().item()
                FP += ((outputs == ones) & (labels == zeros)).sum().item()
                TN += ((outputs == zeros) & (labels == zeros)).sum().item()
        P, R = TP / (TP + FP), TP / (TP + FN)
        F2 = 5 * P * R / (P * 4 + R)
        print(self.model_name, 'F2', F2, 'Precison:', P, 'Recall:', R)
        return F2

    def inference(self, models_name, fold_list):
        # 载入测试集数据
        loader_test = DataLoader(dataset=self.dataset_test, batch_size=128, shuffle=False, num_workers=8)
        test_id_data = pd.read_csv(self.data_dir + "sample_submission.csv")
        threshold_list = []  # 用于构造界的tensor
        for i in range(5):
            threshold_list += [self.threshold[i]] * (self.bound[i + 1] - self.bound[i])  # len ：3474
        threshold_tensor = torch.Tensor([threshold_list] * len(self.dataset_test)).to(
            self.device)  # len(test)*num_classes

        # 开始测试
        preds = torch.zeros((len(self.dataset_test), 3474)).to(self.device)  # 保存计算结果
        for model_name in models_name:
            model = self.model_choice(model_name)  # 选择模型
            if model_name == 'se_resnet50':
                data_dir = '../input/se-resnet50/'
            if model_name == 'se_resnext50':
                data_dir = '../input/se-resnext50/'
            for fold in fold_list:
                single_preds = None  # 记录单模型的单折结果
                parameter_path = data_dir + model_name + str(fold) + '.pth'  # 加载对应折训练参数
                model.load_state_dict(torch.load(parameter_path))
                for (i, images) in enumerate(loader_test):
                    images = images.to(self.device)
                    with torch.no_grad():
                        y_preds = model.forward(images)
                        if i == 0:
                            single_preds = torch.sigmoid(y_preds)  # 第一次
                        else:
                            single_preds = torch.cat((single_preds, torch.sigmoid(y_preds)), dim=0)  # 在0维度上拼接
                        if(i%100==0):
                            print("tick")
                preds += single_preds  # 做累加
                print("done")
            preds /= len(fold_list)  # 每折的平均结果
        preds /= len(models_name)  # 每个模型的平均结果

        # 测试结束

        # 生成submission.csv
        #predictions = preds.to(self.device) > threshold_tensor  # 拼接成0/1矩阵
        # 替换原sample_submission.csv中的attribute_ids列
        #for (i, row) in enumerate(predictions.to('cpu').numpy()):
        #    ids = np.nonzero(row)[0]  # 把0/1矩阵的每一行变为非零元素的索引值数组
        #    test_id_data.iloc[i].attribute_ids = ' '.join([str(x) for x in ids])  # 空格连接，替换列
        #test_id_data.to_csv('submission.csv', index=False)  # 导出csv
        #test_id_data.head()  # 输出csv的前4行

    def model_choice(self, model_name):
        if model_name == 'se_resnet50':
            return se_resnet50(num_classes=3474).to(self.device)
        if model_name == 'se_resnext50':
            return se_resnext50(num_classes=3474).to(self.device)
        if model_name == 'resnext50':
            return resnext50(num_classes=3474).to(self.device)
        if model_name == 'airnext50':
            return airnext50(num_classes=3474).to(self.device)
        if model_name == 'se_resnet101':
            return se_resnet101(num_classes=3474).to(self.device)
        if model_name == 'se_resnext101':
            return se_resnext101(num_classes=3474).to(self.device)
        if model_name == 'resnext101':
            return resnext101(num_classes=3474).to(self.device)

    # 实现在train上对stacking特征的输出,用多个model_name
    def predict(self):
        train_id_data, test_id_data = self.get_id_data()  # 得到id数据，ndarray，用于分折预测train的结果
        preds = None
        train_path = self.data_dir + 'train/'  # 得到数据路径
        for model_name in ['resnet101', 'se_resnext50', 'se_resnext101']:  # 三个模型
            current_preds = None  # 记录单模型的拼接结果
            model = self.model_choice(model_name).to(self.device)  # 选择模型
            for fold in range(6):  # 根据fold选择测试集以及模型参数
                if fold == 5:
                    valid_id_data = train_id_data[23680 * 5:]
                else:
                    valid_id_data = train_id_data[23680 * fold: 23680 * (fold + 1)]  # 得到valid集合
                dataset_valid = MyDataset_stack1(df_data=valid_id_data, mode='train', data_dir=train_path,
                                                 transform=trans_valid)
                loader_valid = DataLoader(dataset=dataset_valid, batch_size=128, shuffle=False,
                                          num_workers=8)  # 得到batch化的数据，用128做为batch合并次数减少
                parameter_path = self.data_dir + model_name + str(fold) + '.pth'  # 根据模型和fold选择模型参数
                model.load_state_dict(torch.load(parameter_path))

                # 进行分折预测并拼接结果
                with torch.no_grad():  # 避免梯度积累
                    for iteration, (images, labels) in enumerate(loader_valid):
                        images = images.to(self.device)  # 放在gpu上,此时的size是batch*5*channels*height*width
                        labels = labels.to(self.device)  # 放在gpu上
                        outputs = model.forward(images)  # shape:batch*3474,用对应的model得到结果
                        outputs = torch.sigmoid(outputs)  # sigmoid非线性激活
                        outputs *= 0.7  # 继承0.7的teacher数据
                        outputs = torch.max(outputs, labels)  # 再与labels取最大
                        if iteration == 0 and fold == 0:
                            current_preds = outputs  # 第一折，第一次
                        else:
                            current_preds = torch.cat((current_preds, outputs), dim=0)  # 在第0维度上拼接

            # 此时current_preds应当为单模型的预测结果
            print(current_preds.size())  # 打印规模结果,应当是14w*3474
            if preds == None:
                preds = current_preds  # 14w*3474
            else:
                preds = torch.cat((preds, current_preds), dim=1)  # 在第1维度上拼接：14w*10422
        preds = torch.transpose(preds, dim0=0, dim1=1)  # 转置，以方便pandas读取

        print(preds.size())
        preds = preds.to('cpu').numpy()
        np.savetxt(self.data_dir + 'train_stack.csv', preds, delimiter=',')

    def analysis(self,  models_name, fold_list):
        # 载入测试集数据
        loader_train = DataLoader(dataset=self.dataset_valid, batch_size=128, shuffle=False, num_workers=8)
        threshold_list = []  # 用于构造界的tensor
        for i in range(5):
            threshold_list += [self.threshold[i]] * (self.bound[i + 1] - self.bound[i])  # len ：3474
        threshold_tensor = torch.Tensor([threshold_list] *500).to(
            self.device)  # len(test)*num_classes

        # 开始测试
        preds = torch.zeros((len(self.dataset_valid), 3474)).to(self.device)  # 保存计算结果
        Label=None
        print("making prediction")
        for model_name in models_name:
            model = self.model_choice(model_name)  # 选择模型
            if model_name == 'se_resnet50':
                data_dir = '../input/se-resnet50/'
                print("modelname: se_resnet50")
            if model_name == 'se_resnext50':
                data_dir = '../input/se-resnext500/'
                print("modelname: se_resnext50")
            if model_name == 'resnet101':
                data_dir = '../input/resnet101/'
                print("modelname: resnet101")
            if model_name == "resnet50":
                data_dir='../input/resnet5007'
            for fold in fold_list:
                start_time=time.time()
                single_preds = None  # 记录单模型的单折结果
                parameter_path = data_dir + model_name + "07" + '.pth'  # 加载对应折训练参数
                model.load_state_dict(torch.load(parameter_path))
                for i, (images, labels) in enumerate(loader_train):
                    if fold==0:
                        if i==0:
                            print("loading labels")
                            Label=labels
                        else:
                            Label=torch.cat((Label,labels),dim=0)
                    images = images.to(self.device)
                    with torch.no_grad():
                        y_preds = model.forward(images)
                        if i == 0:
                            single_preds = torch.sigmoid(y_preds)  # 第一次
                        else:
                            single_preds = torch.cat((single_preds, torch.sigmoid(y_preds)), dim=0)  # 在0维度上拼接

                print("fold %d finished, using time: %d"%(fold,time.time()-start_time))
                start_time=time.time()
                preds+=single_preds
        preds /= len(fold_list)
        preds /= len(models_name)  # 每个模型的平均结果
        predictions = preds.to(self.device) > threshold_tensor
        prediction_list=predictions.to("cpu").numpy().tolist()
        label_list=Label.to("cpu").numpy().tolist()
        pred_list=preds.to("cpu").numpy().tolist()
        different=[]
        for i in range(np.shape(label_list)[0]):
            if label_list[i] != prediction_list[i]:
                count=0
                for j in range(np.shape(label_list)[1]):
                    if label_list[i][j] !=prediction_list[i][j]:
                        count+=1
                different.append([i,count])
        label_csv = pd.DataFrame(data=label_list)
        label_csv.to_csv("./label.csv", mode="w", index=None, header=None)
        print("lable_complete")
        pred_csv = pd.DataFrame(data=pred_list)
        pred_csv.to_csv("./" + model_name + ".csv", mode="w", index=None, header=None)
        print("pred_complete")
        diff_csv = pd.DataFrame(data=different)
        diff_csv.to_csv("./different_" + model_name + ".csv", mode="w", index=None, header=None)
        print("all complete")

    def getpic(self,IDs):
        pic_dir = '../input/imet-2020-fgvc7/train/'
        id_data, _ = self.get_id_data()
        for id in IDs:
            im_path = os.path.join(pic_dir, id_data[id][0] + '.png')
            im = Image.open(im_path)
            im.save("./"+str(id)+".png", 95)

    def model_choice(self, model_name):
        if model_name == 'se_resnet50':
            return se_resnet50(num_classes=3474).to(self.device)
        if model_name == 'se_resnext50':
            return se_resnext50(num_classes=3474).to(self.device)
        if model_name == 'resnext50':
            return resnext50(num_classes=3474).to(self.device)
        if model_name == 'airnext50':
            return airnext50(num_classes=3474).to(self.device)
        if model_name == 'se_resnet101':
            return se_resnet101(num_classes=3474).to(self.device)
        if model_name == 'se_resnext101':
            return se_resnext101(num_classes=3474).to(self.device)
        if model_name == 'resnext101':
            return resnext101(num_classes=3474).to(self.device)
        if model_name == 'resnet101':
            return resnet101(num_classes=3474).to(self.device)
        if model_name == 'resnet50':
            return resnet50(num_classes=3474).to(self.device)

    def ext_spe_att(self, IDS, model_name, fold_list):
        loader_train = DataLoader(dataset=self.dataset_valid, batch_size=128, shuffle=False, num_workers=8)
        print("making prediction")
        preds=None
        model = self.model_choice(model_name)  # 选择模型
        if model_name == 'se_resnet50':
            data_dir = '../input/se-resnet50/'
            print("modelname: se_resnet50")
        if model_name == 'se_resnext50':
            data_dir = '../input/se-resnext500/'
            print("modelname: se_resnext50")
        if model_name == 'resnet101':
            data_dir = '../input/resnet101/'
            print("modelname: resnet101")
        if model_name == "resnet50":
            data_dir = '../input/resnet5007'
            print("model: resnet50")
        for fold in fold_list:
            start_time = time.time()
            single_preds = None  # 记录单模型的单折结果
            parameter_path = data_dir + model_name + str(fold) + '.pth'  # 加载对应折训练参数
            model.load_state_dict(torch.load(parameter_path))
            for i, (images, labels) in enumerate(loader_train):
                images = images.to(self.device)
                with torch.no_grad():
                    y_preds = model.spe_attention(images)
                    if i == 0:
                        single_preds = torch.sigmoid(y_preds)  # 第一次
                    else:
                        single_preds = torch.cat((single_preds, torch.sigmoid(y_preds)), dim=0)  # 在0维度上拼接
            if fold==0:
                preds=single_preds
            else:
                preds=single_preds+preds
            print("fold %d finished, using time: %d" % (fold, time.time() - start_time))
            start_time = time.time()
        preds /= len(fold_list)
        x = preds[:, 0, :, :]
        feature = x.data.to('cpu').numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.round(feature * 255)
        print(feature[0])
        return

num_classes = 3474
num_epoches = 15
batch_size = 32
learning_rate = 2e-4
valid_fold = 3
threshold = [0.1] * 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'se_resnet50'
data_dir = '../input/imet-2020-fgvc7/'
weight_dacay = 1.5e-7

stack1 = Stack1(num_classes=num_classes, lr=learning_rate, device=device, num_epoches=num_epoches,
                batch_size=batch_size, threshold=threshold, model_name=model_name, data_dir=data_dir,
                valid_fold=valid_fold, weight_dacay=weight_dacay)
stack1.ext_spe_att([0], 'se_resnet50', [0,1,2,3,4,5])
#stack1.inference(models_name=['se_resnet50', 'se_resnext50'], fold_list=[0, 1, 2, 3, 4,5])
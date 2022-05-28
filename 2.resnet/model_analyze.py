import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 定义18层和34层的残差结构
    expansion = 1  # 残差结构中主分支结构（卷积和个数）是否有变化，这里18层和34层主分支无变化设置为1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample：虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 采用BN，不使用偏置
        self.bn1 = nn.BatchNorm2d(out_channel)  # 进行BN
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):  # 定义正向传播，包含残差部分
        identity = x
        if self.downsample is not None:  # None代表实线残差结构，虚线代表虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 添加捷径输出
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # 定义50层101层和152层的残差结构
    expansion = 4  # 50，101，152层中，conv2_x等中，卷积核个数变化为4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)  # 开始BN
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,  # 上述的第3层变化
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):  # 定义正向传播过程
        identity = x
        if self.downsample is not None:  # None代表实线残差结构，虚线代表虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):  # 定义ResNet整个网络框架

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # max pool之后得到的输入特征矩阵深度

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 定义第二层，即max pool
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # conv2_x对应的一系列残差结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv3_x对应的一系列残差结构
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv4_x对应的一系列残差结构
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv5_x对应的一系列残差结构
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1) 自适应的平均池化下采样
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 定义全连接层

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []  # 定义空的列表，嵌入虚线残差结构
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):  # 嵌入实线残差结构
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)  # 通过非关键字参数的形式传入nn.Seq中，组合在一起并返回

    def forward(self, x):
        outputs = []  # 设置空列表，获取中间层的输出特征矩阵
        x = self.conv1(x)
        outputs.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outputs.append(x)

        return outputs


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

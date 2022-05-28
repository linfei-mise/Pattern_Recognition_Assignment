import torch.nn as nn
import torch


# 创建类：AlexNet，继承nn.Module这个父类
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):  # 初始化函数定义网络在正向传播过程中要用到的层结构
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # Sequential将以下结构进行打包处理，构成新的结构，减少工作量。features：提取图像特征的结构
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),  # inplace:pytorch中增加计算量，但降低内存使用量的方法
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27] stride默认为1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(  # 全连接层
            nn.Dropout(p=0.5),  # 全连接层节点按照一定比例失活，防止过拟合
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:  # 初始化权重，pytorch中默认采用 kaiming_normal 进行初始化权重
            self._initialize_weights()

    def forward(self, x):  # 定义正向传播过程
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平处理，从高和宽维度展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 返回一个迭代器，遍历模型中的所有模块
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 如果偏置不为空
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 通过正态分布给权重进行赋值，均值0，方差0.01
                nn.init.constant_(m.bias, 0)

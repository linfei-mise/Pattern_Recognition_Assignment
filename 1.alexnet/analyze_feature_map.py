import torch
from model_analyze import AlexNet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(  # 图像预处理过程，要与训练过程中的预处理过程保持一致
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 实例化AlexNet模型
model = AlexNet(num_classes=2)
# 载入训练好的模型参数（权重文件）
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)  # 打印模型结构

# 载入想要预测的图片，并进行预处理
img = Image.open("plot_img/1.jpg")
# [N, C, H, W]
img = data_transform(img)
# 对这张图片扩充一个维度，进而将图片输入到模型中进行预测
img = torch.unsqueeze(img, dim=0)

# forward，将图片输入到模型中进行正向传播
out_put = model(img)  # model_analyze文件中的out_put,即三个卷积层结构特征矩阵列表
for feature_map in out_put:  # 遍历每个卷积层的输出特征矩阵
    # [N, C, H, W] -> [C, H, W]，只输入一张图片所以维度N，也就是Bitch维度是无意义的
    im = np.squeeze(feature_map.detach().numpy())  # 将得到的feature_map转化为numpy格式，方便后续分析，压缩掉bitch维度
    # [C, H, W] -> [H, W, C] 即 [0,1,2] -> [1,2,0]
    im = np.transpose(im, [1, 2, 0])  # 转换特征矩阵维度排列顺序，变为常规的图片维度格式

    # 打印12张特征图
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)  # 3即绘制图片的行数，4即绘制图片的列数，i+1即图片对应的索引
        # [H, W, C] 如果添加cmap='gray'，即以灰度图像的形式进行展示，这里采用蓝色和绿色混合图像进行展示，视觉效果更好看一些
        plt.imshow(im[:, :, i])  # 展示特征图信息，通过python切片的形式获取对应的channel,即一个二维特征矩阵
    plt.show()

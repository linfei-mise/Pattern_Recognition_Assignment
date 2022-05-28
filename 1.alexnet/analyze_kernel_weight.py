import torch
from model_analyze import AlexNet
import matplotlib.pyplot as plt
import numpy as np


# 实例化模型
model = AlexNet(num_classes=2)
# 载入模型参数
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))  # 可以不用实例化模型，可以直接通过torch.load函数加载权重文件
print(model)

weights_keys = model.state_dict().keys()  # 获取模型中所有的可训练参数(卷积层）的字典(dict),通过keys获取层结构的名称
for key in weights_keys:
    # 排除BN层结构中的不需要的信息
    if "num_batches_tracked" in key:
        continue
    # 传入参数[kernel_number, kernel_channel, kernel_height, kernel_width]，并转换为numpy的格式
    weight_t = model.state_dict()[key].numpy()

    # 可以以切片的方式获取特定卷积层的信息，这里选择使用全部卷积层的信息
    # k = weight_t[0, :, :, :]

    # calculate mean, std, min, max
    weight_mean = weight_t.mean()  # 计算均值
    weight_std = weight_t.std(ddof=1)  # 计算标准差
    weight_min = weight_t.min()  # 计算最小值
    weight_max = weight_t.max()  # 计算最大值
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1])  # 将卷积核的权重展成一维的向量
    plt.hist(weight_vec, bins=50)  # 统计绘制直方图，将最小值和最大值之间均分成50等份
    plt.title(key)
    plt.show()


import torch
from model_analyze import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms


data_transform = transforms.Compose(  # 与Resnet网络训练过程图像预处理保持一致
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 实例化Resnet网络模型
model = resnet34(num_classes=2)
# 载入模型权重
model_weight_path = "./resNet34.pth"  # 载入resNet34，也即我们训练好的权重信息
model.load_state_dict(torch.load(model_weight_path))
print(model)

# 导入预测的图片
img = Image.open("plot_img/1.jpg")
# [N, C, H, W]
img = data_transform(img)
# 扩充一个维度
img = torch.unsqueeze(img, dim=0)

# 定义正向传播过程
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # 打印12张图片
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i])
    plt.show()


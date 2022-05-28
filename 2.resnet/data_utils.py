import os
import json
import pickle
import random

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_class_preds(net,  # 实例化的模型
                     images_dir: str,  # plot_img根目录
                     transform,  # 验证集使用的图像预处理
                     num_plot: int = 5,  # 总共要展示的图片数目
                     device="gpu"):
    if not os.path.exists(images_dir):  # 判断文件根目录是否存在
        print("not found {} path, ignore add figure.".format(images_dir))
        return None

    label_path = os.path.join(images_dir, "label.txt")  # 判断是否有label.txt文件
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # read class_indict
    json_label_path = './class_indices.json'  # 生成json文件
    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    json_file = open(json_label_path, 'r')
    # {"0": "No_hat"}
    hat_class = json.load(json_file)
    # {"No_hat": "0"}
    class_indices = dict((v, k) for k, v in hat_class.items())  # 对换位置

    # reading label.txt file
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():  # 读取每行信息
            line = line.strip()  # 去除每行首尾的空格以及换行符
            if len(line) > 0:  # 判断字符长度是否大于零
                split_info = [i for i in line.split(" ") if len(i) > 0]  # 按空格进行分割
                assert len(split_info) == 2, "label format error, expect file_name and class_name"
                image_name, class_name = split_info
                image_path = os.path.join(images_dir, image_name)  # 得到绝对路径
                # 如果文件不存在，则跳过
                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue
                # 如果读取的类别不在给定的类别内，则跳过
                if class_name not in class_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                label_info.append([image_path, class_name])

    if len(label_info) == 0:
        return None

    # get first num_plot info
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images = []  # 批量预测图片
    labels = []
    for img_path, class_name in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")  # 如果图片不是RGB，转化为RGB格式
        label_index = int(class_indices[class_name])

        # preprocessing
        img = transform(img)  # 验证机预处理方法
        images.append(img)
        labels.append(label_index)

    # batching images
    images = torch.stack(images, dim=0).to(device)

    # inference
    with torch.no_grad():  # 禁止跟踪梯度
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=100)  # w=250,h=300,dpi=100
    for i in range(num_imgs):
        # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
        ax = fig.add_subplot(1, num_imgs, i+1, xticks=[], yticks=[])

        # CHW -> HWC
        npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255  # 图片范围0到255
        plt.imshow(npimg.astype('uint8'))

        title = "{}, {:.2f}%\n(label: {})".format(  # 构建展示的标签
            hat_class[str(preds[i])],  # predict class
            probs[i] * 100,  # predict probability
            hat_class[str(labels[i])]  # true class
        )
        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))  # 通过颜色直观表示

    return fig

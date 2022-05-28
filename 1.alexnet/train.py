import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from data_utils import plot_class_preds
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="runs/hat_experiment")  # tensorboard文件保存位置

    data_transform = {  # 数据预处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 水平方向随即翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取当前train文件目录
    image_path = os.path.join(data_root, "0.data")  # 加上0.data路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),  # 载入训练集数据
                                         transform=data_transform["train"])  # 数据预处理
    train_num = len(train_dataset)  # 打印训练集有多少张图片

    # "0": "No_hat"  "1": "With_hat"
    hat_list = train_dataset.class_to_idx  # 获取分类名称对应的索引，hat_list
    cla_dict = dict((val, key) for key, val in hat_list.items())  # 创建字典
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=2)  # indent: 参数根据数据格式缩进显示，读起来更加清晰，这里2分类设置为2
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)  # windows下只能选择为0，不能指定加载线程数目

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = AlexNet(num_classes=2, init_weights=True)  # 传入AlexNet
    net.to(device)  # 网络规定在设备上，这里默认到GPU上

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 224, 224), device=device)  # 创建零矩阵，参数与模型输入参数相同
    tb_writer.add_graph(net, init_img)

    loss_function = nn.CrossEntropyLoss()  # 针对多类别的损失交叉熵函数
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 定义优化器Adam，学习率0.0002

    epochs = 100
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # 管理Dropout是否使用，这里启用
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()  # 清空梯度信息
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()  # 反向传播
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 验证过程中不会计算损失梯度
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        tags = ["train_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], loss, epoch)
        tb_writer.add_scalar(tags[1], val_accurate, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # add figure into tensorboard
        fig = plot_class_preds(net=net,
                               images_dir="./plot_img",
                               transform=data_transform["val"],
                               num_plot=5,
                               device=device)
        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()

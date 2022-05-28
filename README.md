## 该文件夹是用来存放本学期模式识别课程作业——代码部分的目录
### 模式识别课程作业要求如下：
#### 分組報告內容
* （1）分組報告數據會有另外文件(人臉數據集）上傳到群裡，大家之後自行下載【Masked-Face-Dataset.zip】。
* （2）解壓後自己選擇兩種分類去做，比如分男女、分是否带口罩、 是否带帽子等。
* （3） 每一種分類（特徵）用4~5種方法（算法）。
* （4） 報告提交需含有結果、分析(如優缺點、準確度等、源代碼（可以使用開源包，代碼加注釋）。
#### 分組報告提交日期：5月底前
### 完成情况如下：
* （1）这里选择的特征是帽子，即通过给的人脸数据集训练模型，使得模型可以准确的预测任何图片上的人是否带帽子。
* （2）4种算法选择的均为深度学习算法。
* （3）算法网络框架包括alexnet、resnet、vision_transformer、convnext。
* （4）在alexnet网络中采用传统的网络训练方式。
* （5）在resnet网络中采用传统的网络训练方式与加入迁移学习两种方式，并进行了实验对比分析。
* （6）在vision_transformer网络中采用迁移学习的方式训练网络。
* （7）在convnext网络中采用传统的网络训练方式与加入迁移学习两种方式，并进行了实验对比分析。
* （8）模型的优缺点、准确度等分析详情，请见代码文件目录以及论文报告。
* （9）每个算法文件中均有README.md文件，介绍了每个文件夹及文件的用途。
* （10）每个代码脚本中，重要的位置均写有注释，方便查看。
### 模型特点如下：
* （1）在resnet网络和convnext网络引入迁移学习的概念，并通过实验对比分析迁移学习与传统训练方式的优缺点。
* （2）使用vision_transformer架构搭建网络，对比分析self-attention机制与传统的卷积神经网络的优缺点。
* （3）对于alexnet、resnet、convnext网络分析了部分重要卷积层的权重分布。
* （4）对于alexnet、resnet网络输出了部分重要的卷积层的feature_map。
* （5）由于vision_transformer模型在Patch embedding部分将输入特征矩阵展成二维，这里并没有对attention机制进行还原可视化feature_map。
* （6）对于vision_transformer模型，通过Grad-CAM的方法进行attention的可视化。
* （7）对于alexnet、resnet、vision_transformer、convnext模型均输入了混淆矩阵，并计算了相应的模型指标。
* （8）所有的训练结果均通过tensorboard进行了可视化，可随时调用查看，分析更加直观，并保证了模型训练的真实性。
### 训练设备信息如下：
* （1）处理器：HexaCore AMD Ryzen 5 5600X, 4650 MHz (46.5 x 100)
* （2）内存：Crucial Ballistix 48 GB (8 GB × 2 + 16 GB × 2)
* （3）显卡：NVIDIA GeForce RTX 3070 Ti  (8 GB)
### PyTorch深度学习框架使用版本介绍如下：
* （1）torch-1.9.0+cu111-cp38-cp38-win_amd64
* （2）torchvision-0.10.0+cu111-cp38-cp38-win_amd64
* （3）其他Python库由于比较多，这里就不做过多赘述，请根据代码中import部分自行安装。
### 代码提交方式如下：
* （1）本次代码提交采用Github和Google Drive两种方式。
* （2）代码将上传到Github中开源，其中训练权重文件由于Github文件上传大小限制，将上传Google Drive，相应的权重文件链接请点击下方获取。
* （3）在论文中将会附有相应的Github链接。
### 权重文件下载地址如下：
* （1）alexnet网络训练得到权重文件：[AlexNet.pth](https://drive.google.com/file/d/1qyTeYHcE2Ybm5xCh3b5Zf4MxyHjknLC2/view?usp=sharing)
* （2）resnet网络不使用迁移学习训练得到的权重文件：[resnet_ntl_best.pth](https://drive.google.com/file/d/14KxcMyVs9PllQAaUxgOYB6GC8V8rQ4Zv/view?usp=sharing)
* （3）resnet网络使用迁移学习所用的预训练权重文件：[resnet34-pre.pth](https://drive.google.com/file/d/1jLPvgFgLvii1a435_oIqnD2GI-CrB8E1/view?usp=sharing)
* （4）resnet网络使用迁移学习训练得到的权重文件：[resNet34.pth](https://drive.google.com/file/d/1W4XSOet_41H4dhTl4wMzjjPsg1sJ1xDA/view?usp=sharing)
* （5）vision_transformer网络使用迁移学习所用的预训练权重文件：[vit_base_patch16_224_in21k.pth](https://drive.google.com/file/d/1psvtt5iSWINWk6ePg5K-vTmVAXxU9JRZ/view?usp=sharing)
* （6）vision_transformer网络使用迁移学习训练得到的权重文件：[ViT_best.pth](https://drive.google.com/file/d/1bZQ38N9Ys6uK82MOQt02r4WqDJ2S1821/view?usp=sharing)
* （7）convnext网络不使用迁移学习训练得到的权重文件：[ntl_best_model.pth](https://drive.google.com/file/d/1k9z3dhlL6wWEECMrW37ZQxsTJTEKxz7j/view?usp=sharing)
* （8）convnext网络使用迁移学习所用的预训练权重文件：[convnext_base_22k_224.pth](https://drive.google.com/file/d/1hsDAW80OdbJCjzRnCl258QQwjxiWrDJR/view?usp=sharing)
* （9）convnext网络使用迁移学习训练得到的权重文件：[best_model.pth](https://drive.google.com/file/d/1jqrwSbBj3Lg24IzzsAd2pYRARkrg8Y7v/view?usp=sharing)
### 本目录文件夹介绍如下：
```
├── 0.data（存放训练数据的位置）
├── 1.alexnet（alexnet网络代码存放位置）
├── 2.resnet（resnet网络代码存放位置）
├── 3.vision_transformer（vision_transformer网络代码存放位置）
├── 4.convnext（convnext网络代码存放位置）
└── README.md（作业完成情况须知）
```

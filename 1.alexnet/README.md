## 该文件夹是用来存alexnet模型文件的目录
### 下面将针对子文件夹及模型文件进行简单的介绍： 
* （1）文件夹
```
├── conv_feature_map_results（analyze_feature_map.py脚本生成的部分特征图）
├── conv_kernel_weight_results（analyze_kernel_weight.py脚本生成的部分权重、偏置等直方图）
├── plot_img（传入TensorBoard进行可视化的图片，以及predict.py脚本单次预测的图片路径）
├── Post_experimental_processing（存放生成混淆矩阵及模型指标的脚本及图片）
├── runs（传入TensorBoard进行可视化的参数文件，通过train.py脚本生成）
└── tensorboard_results（TensorBoard可视化得到的结果图）
```
* （2）文件
``` 
├── AlexNet.pth（alexnet模型训练得到的权重文件）  
├── analyze_feature_map.py（生成特征图所应用的脚本）  
├── analyze_kernel_weight.py（生成直方图所应用的脚本）
├── class_indices.json（生成的json文件用于TensorBoard和predict.py中类别可视化索引）
├── data_utils.py（构建TensorBoard可视化的脚本）
├── model.py（alexnet模型文件）
├── model_analyze.py（alexnet模型分析文件，更改了正向传播过程，用于生成特征图和直方图）
├── predict.py（模型的预测脚本）
├── README.md（文件类型解释）
└──  train.py（模型训练脚本）
```

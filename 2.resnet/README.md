## 该文件夹是用来存resnet模型文件的目录
### 下面将针对子文件夹及模型文件进行简单的介绍： 
* （1）文件夹
```
├── conv_feature_map_results（analyze_feature_map.py脚本生成的部分特征图）
├── conv_kernel_weight_results（analyze_kernel_weight.py脚本生成的部分权重、偏置等直方图）
├── plot_img（传入TensorBoard进行可视化的图片，以及predict.py脚本单次预测的图片路径）
├── Post_experimental_processing（存放生成混淆矩阵及模型指标的脚本及图片）
├── resnet_no_transfer_learning（不使用迁移学习训练模型的文件夹）
├── runs（传入TensorBoard进行可视化的参数文件，通过train.py脚本生成）
└── tensorboard_results（TensorBoard可视化得到的结果图）
```
* （2）文件夹 -- 不使用迁移学习训练的文件夹resnet_no_transfer_learning
```
├── ntl_runs（传入TensorBoard进行可视化的参数文件，通过train_ntl.py脚本生成）
├── plot_img_ntl（传入TensorBoard进行可视化的图片）
├── Post_experimental_processing_ntl（存放生成混淆矩阵及模型指标的脚本及图片）
├── tensorboard_results_ntl（TensorBoard可视化得到的结果图）
└── weights（迭代过程中记录的每一次迭代过程的权重文件）
```
* （3）文件 -- 不使用迁移学习训练的文件夹下的文件
```
├── class_indices.json（生成的json文件用于TensorBoard）
├── data_utils_ntl.py（定义训练集&验证集划分方法，以及写入TensorBoard可视化方法的脚本）
├── model_ntl.py（不使用迁移学习的resnet模型文件）
├── my_dataset_ntl.py（自定义数据集的方法脚本）
├── train_eval_utils_ntl.py（多GPU并行计算脚本，这里采用单GPU）
└── train_ntl.py（不使用迁移学习的resnet模型训练脚本）
```
* （4）文件 -- 使用迁移学习训练
``` 
├── analyze_feature_map.py（生成特征图所应用的脚本）  
├── analyze_kernel_weight.py（生成直方图所应用的脚本）
├── batch_predict.py（批量预测脚本）
├── class_indices.json（生成的json文件用于TensorBoard和predict.py中类别可视化索引）
├── data_utils.py（构建TensorBoard可视化的脚本）
├── load_weights.py（使用迁移学习预训练权重载入脚本）
├── model.py（resnet模型文件）
├── model_analyze.py（resnet模型分析文件，更改了正向传播过程，用于生成特征图和直方图）
├── predict.py（模型的预测脚本）
├── README.md（文件类型解释）
├── resNet34.pth（使用迁移学习训练得到的本训练集的权重文件）
├── resnet34-pre.pth（官方下载的预训练模型权重文件）
└── train.py（模型训练脚本）
```

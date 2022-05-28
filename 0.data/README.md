## 该文件夹是用来存放训练数据的目录
### 关于数据集的处理步骤如下：
* （1）原始数据集为老师给的Masked-Face-Dataset / masked_whn。
* （2）将原始数据集中的所有图片从按名字划分的文件夹中提取出来。
* （3）将提取后的数据集进行打标签并根据标签进行文件夹划分。
* （4）得到的文件夹即本目录下的Dataset_changes_2中的No_hat和With_hat。
* （5）编写split_hat_data.py脚本。
* （6）执行"split_hat_data.py"脚本自动将数据集划分成训练集train和验证集val。
* （7）执行方式需通过Windows PowerShell或Anaconda Powershell Prompt运行该脚本。
* （8）训练集train和验证集val按照9：1的比例划分。
* （9）hat_data_number_from_resnet.png 文件统计了训练集和验证集图片的数量。

```
├── 0.data   
       ├── Dataset_changes_2（masked_whn数据集提取出的文件夹，4,151个样本）  
       ├── train（生成的训练集，3,737个样本）  
       ├── val（生成的验证集，414个样本）
       └── split_hat_data.py（数据集划分脚本） 
```

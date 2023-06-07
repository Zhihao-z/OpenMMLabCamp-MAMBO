OpenMMLabCamp-MAMBO
Warehouse for submitting tasks in OpenMMLabCamp

题目：基于 ResNet50 的水果分类
背景：使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类
任务：
划分训练集和验证集
按照 MMPreTrain CustomDataset 格式组织训练集和验证集
使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型
在水果数据集上进行微调训练
使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行 分类
需提交的验证集评估指标（不能低于 60%）

1、准备数据集

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/f712b7d5-3a8b-4cf3-ae8b-9b7393b09b59)

按7:3划分

2、训练水果分类模型

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/478fd8ac-0155-431f-8989-914603cf28e4)

配置文件在data/resnet50_fruit30.py

3、训练结果

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/ab9b48ea-a413-4a6e-8c4b-09d1d2e2851c)

accuracy/top1: 90.6344  accuracy/top5: 99.3202  data_time: 0.0017  time: 0.0255

4、可视化

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/6d8b290b-6990-4dce-9ba5-f72085f1f283)

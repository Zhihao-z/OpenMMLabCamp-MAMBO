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

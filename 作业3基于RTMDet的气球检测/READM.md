OpenMMLabCamp-MAMBO
Warehouse for submitting tasks in OpenMMLabCamp

作业：基于 RTMDet 的气球检测

背景：熟悉目标检测和 MMDetection 常用自定义流程。

任务：

基于提供的 notebook，将 cat 数据集换成气球数据集
按照视频中 notebook 步骤，可视化数据集和标签
使用MMDetection算法库，训练 RTMDet 气球目标检测算法，可以适当调参，提交测试集评估指标
用网上下载的任意包括气球的图片进行预测，将预测结果发到群里
按照视频中 notebook 步骤，对 demo 图片进行特征图可视化和 Box AM 可视化，将结果发到群里
需提交的测试集评估指标（不能低于baseline指标的50%）
目标检测 RTMDet-tiny 模型结果的 mAP 不低于 65
数据集
气球数据集可以直接下载 https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip


1、处理生成coco的json

[balloon_train.json](./balloon_train.json)

[balloon_val.json](./balloon_val.json)

2、自定义配置文件

[rtmdet_tiny_1xb12-40e_balloon.py](./rtmdet_tiny_1xb12-40e_balloon.py)

3、训练前可视化

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/2b6e8353-dec3-4034-af58-d4562bea7f36)

4、模型训练
```
!python tools/train.py rtmdet_tiny_1xb12-40e_balloon.py
```
5、训练结果：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/26142134-2e03-41d5-8ab2-e55b24b951a6)

6、val集预测可视化：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/d1b30522-b72b-4187-ab9a-02d47f6b5497)

7、网络气球图像预测：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/8d9d7807-e991-48db-85df-1802c9d754d7)

8、特征图可视化：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/9fe04912-a884-4943-943a-72cfa481582e)

9、Box AM可视化：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/b13925ac-d1ec-4dfb-b11a-042de58337b0)


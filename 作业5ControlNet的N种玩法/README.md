作业：
MMSeg 语义分割
背景：
西瓜瓤、西瓜皮、西瓜籽像素级语义分割
TO DO LIST：
Labelme 标注语义分割数据集（子豪兄已经帮你完成了）；
划分训练集和测试集（子豪兄已经帮你完成了）；
Labelme 标注转 Mask 灰度图格式（子豪兄已经帮你完成了）；
使用 MMSegmentation 算法库，撰写 config 配置文件，训练 PSPNet 语义分割算法；
提交测试集评估指标；
自己拍摄西瓜图片和视频，将预测结果发到群里；
（选做）训练 Segformer 语义分割算法，提交测试集评估指标。


改数据集配置文件,放到mmseg/datasets下面：
watermelon.py
```
# 参考cityscapes.py格式

# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WatermelonDataset(BaseSegDataset):
    """Watermelon dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for Watermelon dataset.
    """
    METAINFO = dict(
        classes=('red', 'green', 'white', 'seed-black', 'seed-white', 'Unlabeled'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
```

撰写 config 配置文件：
pspnet-watermelon.py

模型训练：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/be673cb5-aa5e-4761-800d-f5124de5fed0)


训练结果：

![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/80707dda-362b-497a-871a-10b984964c61)

原图：
![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/351d32bc-521c-43c6-9e77-fbf53c6747a5)


可视化：
![image](https://github.com/MAMOB/OpenMMLabCamp-MAMBO/assets/42363751/7291c30b-cc2e-4695-a2f4-73eea88b447d)






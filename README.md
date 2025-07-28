# RG-YOLO: Efficient and real-time Aerial Object Detection integrating RepVGG and GhostConv

### 简介

针对无人机红外图像空中目标检测（ODAI）在边缘设备上面临的挑战，我们提出了RG-YOLO，一个基于改进YOLOv11s的轻量级模型。

- 通过在骨干网络中集成改进的RepVGG（将ReLU替换为SiLU），并引入GhostConv到C3K2模块（形成C3Ghost），显著优化了模型效率和红外数据特征提取能力，尤其适用于低对比度红外场景。
- 在ComNet、HIT-UAV和FLIR红外数据集上进行了实验验证，我们的模型实现了**453.43** FPS的速度（相比基线提升**52.5**%），同时在保持接近原始检测精度的前提下，相比YOLOv11s和YOLOv13s，参数量、计算量和模型大小均大幅降低（例如，参数量最高降低**76.7**%，计算量最高降低**70**%，模型大小最高降低**75.3**%），实现了更优的速度-精度权衡。
- 本算法高度适用于资源受限的边缘设备，增强实时空中图像监控，尤其是在挑战性红外环境中。

## 快速使用

### 安装

```shell
git clone https://github.com/yhlyhlyhlyhlyhl/RG-YOLO.git
cd RG-YOLO
pip install -r requirements.txt
```

### 结构文件

```
ultralytics/cfg/models/RG-YOLO/yolo11-rep-ghost.yaml
```

### 训练

```
from ultralytics.models import YOLO
import os


if __name__ == '__main__':
model = YOLO(model='ultralytics/cfg/models/RG-YOLO/yolo11-rep-ghost.yaml')
model.train(data='./data.yaml', epochs=2, batch=1, device='0', imgsz=640, workers=2, cache=False, amp=True, mosaic=False, project='runs/train', name='exp')
```



### 引用RG-YOLO

```
Not updated yet.
```

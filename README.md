# AnyOCR

```
    ___                ____  __________ 
   /   |  ____  __  __/ __ \/ ____/ __ \
  / /| | / __ \/ / / / / / / /   / /_/ /
 / ___ |/ / / / /_/ / /_/ / /___/ _, _/ 
/_/  |_/_/ /_/\__, /\____/\____/_/ |_|  
             /____/                     

```

简体中文 | [English](./README_en.md)


## 1. 简介

目前，我们非常开心的推出了兼容多平台的onnx格式的ocr工具`AnyOCR`，其核心亮点在于采用ONNXRuntime作为推理引擎，相比PaddlePaddle推理引擎，确保了高效稳定的运行。

- github地址：[AnyOCR](https://github.com/oriforge/anyocr)

## 2. 缘起

PaddlePaddle团队在PaddleOCR项目上，实现了一个基于PaddlePaddle的OCR工具，其性能和功能都十分强大，但是，在某些场景下，PaddlePaddle推理引擎的运行速度和稳定性，都存在一些问题。所以我们搜集很多新的OCR数据对paddleocr进行微调优化，并导出成onnx格式，直接使用onnxruntime推理，避开paddlepaddle推理引擎的坑，并支持cpu，gpu等。

Paddleocr在一些新型的数据上或者领域数据上表现的并不是很好，所以我们采集了很多数据进行微调训练，覆盖各个领域，包括：
- cc-ocr
- 工业
- 医疗
- 体检
- 中文
- 英文
- 论文
- 网络
- 自建
- 等等

数据集总计：大于`385K`。

### 扩展训练

- 训练集：`385K`
- 测试集：`5k`
- 准确率：`0.952`

### 模型介绍

- 检测模型：`anyocr_det_ch_v4_lite.onnx`，由`ch_PP-OCRv4_det`在我们的数据集上微调训练而来。
- 识别模型：`anyocr_rec_v4_server.onnx`，由`ch_PP-OCRv4_server_rec`在我们的数据集上微调训练而来。
- 方向分类：`anyocr_cls_v4.onnx`，来源于`ch_ppocr_mobile_v2.0_cls`未做训练。
- 文字字符：`anyocr_keys_v4.txt`，来源于`ppocr/utils/ppocr_keys_v1.txt`。


### 评估

自建评估集：`1.1K`

抽取1150对未训练的数据作为评估，覆盖中文，英文，数字，符号等。

我们的评估集与其它ocr准确率的测试评估：

 - anyocr: 0.97
 - 百度paddleocr：0.92
 - 阿里通义读光ocr：0.86
 - 阶跃星辰GOT_OCR2.0：0.89
 - olm-ocr: 0.46

## 3. 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

```python
## simple
# use_det = True or False, 是否使用文本检测
# use_cls = True or False, 是否使用文本方向
# use_rec = True or False, 是否使用文本识别

from anyocr.pipeline import anyocr

model = anyocr()

res = model.raw_completions('/to/your/image',use_cls=True,use_det=True)

print(res)


## 返回单字坐标

from anyocr.pipeline import anyocr

model = anyocr()

res = model.raw_completions('/to/your/image',use_cls=True,use_det=True,return_word_box = True)


### 自定义模型地址
from anyocr.pipeline import anyocr
from anyocr.pipeline import anyocrConfig


config = anyocrConfig(
    det_model_path = "anyocr/models/anyocr_det_ch_v4_lite.onnx",
    rec_model_path = "anyocr/models/anyocr_rec_v4_server.onnx",
    cls_model_path = "anyocr/models/anyocr_cls_v4.onnx",
    rec_keys_path = "anyocr/models/anyocr_keys_v4.txt"   
)
config = config.model_dump()
model = anyocr(config)

res = model.raw_completions('/to/your/image',use_cls=True,use_det=True)

print(res)
```

- 如果您有更好的文字检测，文本识别识别也可以只使用我们的一部分。
- 您也可以将paddleocr的模型导出成onnx格式，使用AnyOCR推理，或者您自己微调的paddleocr模型，使用AnyOCR推理。


### 参数配置

```python
from pydantic import BaseModel

class anyocrConfig(BaseModel):
    text_score: float = 0.5   # 文本识别结果置信度，取值范围：[0, 1]
    use_det: bool = True  # 是否使用文本检测
    use_cls: bool = True  # 是否使用文本行方向分类
    use_rec: bool = True  # 是否使用文本行识别
    print_verbose: bool = False # 打印进度
    min_height: int = 30  # 图像最小高度（单位是像素），低于这个值，会跳过文本检测阶段，直接进行后续识别。
    width_height_ratio: float = 8 # 如果输入图像的宽高比大于width_height_ratio，则会跳过文本检测，直接进行后续识别
    max_side_len: int = 2000 #  如果输入图像的最大边大于max_side_len，则会按宽高比，将最大边缩放到max_side_len
    min_side_len: int = 30 # 如果输入图像的最小边小于min_side_len，则会按宽高比，将最小边缩放到min_side_len
    return_word_box: bool = False # 是否返回文字的单字坐标。
    
    det_use_cuda: bool = False  # 是否使用gpu
    det_model_path: Optional[str] = None #文本检测模型路径
    det_limit_side_len: float = 736 # 限制图像边的长度的像素值。
    det_limit_type: str = "min" # 限制图像的最小边长度还是最大边为limit_side_len，取值范围为：[min, max]
    det_max_candidates:int = 1000 # 最大候选框数目
    det_thresh: float = 0.3  # 图像中文字部分和背景部分分割阈值。值越大，文字部分会越小。取值范围：[0, 1]
    det_box_thresh: float = 0.5 # 文本检测所得框是否保留的阈值，值越大，召回率越低。取值范围：[0, 1]
    det_unclip_ratio: float = 1.6 # 控制文本检测框的大小，值越大，检测框整体越大。取值范围：[1.6, 2.0]
    det_donot_use_dilation: bool = False # 是否使用膨胀，该参数用于将检测到的文本区域做形态学的膨胀处理。
    det_score_mode: str = "slow"  # 计算文本框得分的方式。取值范围为：[slow, fast]
    
    cls_use_cuda: bool = False  # 是否使用gpu
    cls_model_path: Optional[str] = None #  文本行方向分类模型路径
    cls_image_shape: List[int] = [3, 48, 192] # 输入方向分类模型的图像Shape(CHW)
    cls_label_list: List[str] = ["0", "180"] # 方向分类的标签，0°或者180°，该参数不能动。
    cls_batch_num: int = 6 # 批次推理的batch大小，一般采用默认值即可，太大并没有明显提速，效果还可能会差。默认值为6。
    cls_thresh: float = 0.9 # 方向分类结果的置信度。取值范围：[0, 1]
    
    rec_use_cuda: bool = False  # 是否使用gpu
    rec_keys_path: Optional[str] = None # 文本识别模型对应的字典文件
    rec_model_path: Optional[str] = None # 文本识别模型路径
    rec_img_shape: List[int] = [3, 48, 320] # 输入文本识别模型的图像Shape(CHW)
    rec_batch_num: int = 6 # 批次推理的batch大小，一般采用默认值即可，太大并没有明显提速，效果还可能会差。默认值为6。

```

## 特别鸣谢
- `paddleocr`提供原始模型以及微调教程
- 大部分源码来源于`RapidOCR`，个人做了一些改动

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=oriforge/anyocr&type=Date)](https://www.star-history.com/#oriforge/anyocr&Date)

# 当前使用的模型信息 / Current Model Information

## 中文版本

### 主要检测模型
**当前使用模型：Grounding DINO with SwinT-OGC**
- 模型类型：多模态目标检测模型（文本+图像）
- 骨干网络：Swin Transformer (SwinT)
- 配置文件：`GroundingDINO_SwinT_OGC.py`
- 预训练权重：`groundingdino_swint_ogc.pth` (~690MB)
- 输入分辨率：640x480
- 性能：单图像检测最快200ms (4080 Super GPU)

### 文本编码模型
**当前使用模型：BERT-base-uncased**
- 用途：处理文本提示词进行语义理解
- 模型大小：约110M参数
- 位置：`local_models/bert-base-uncased/`

### 翻译模型
**当前使用模型：离线中英翻译模型**
- 用途：将中文提示词翻译为英文
- 位置：`offline-zh-en-model/zh-en-model/`
- 支持：中文输入自动翻译为英文

### 可选模型配置
系统还支持以下模型配置（需要相应权重文件）：
- GroundingDINO_SwinB_cfg：使用SwinB骨干网络（更大更精确但更慢）

---

## English Version

### Primary Detection Model
**Currently Used: Grounding DINO with SwinT-OGC**
- Model Type: Multi-modal object detection (Text + Image)
- Backbone: Swin Transformer (SwinT)
- Configuration File: `GroundingDINO_SwinT_OGC.py`
- Pretrained Weights: `groundingdino_swint_ogc.pth` (~690MB)
- Input Resolution: 640x480
- Performance: Fastest 200ms per image (4080 Super GPU)

### Text Encoding Model
**Currently Used: BERT-base-uncased**
- Purpose: Processing text prompts for semantic understanding
- Model Size: ~110M parameters
- Location: `local_models/bert-base-uncased/`

### Translation Model
**Currently Used: Offline Chinese-English Translation Model**
- Purpose: Translate Chinese prompts to English
- Location: `offline-zh-en-model/zh-en-model/`
- Feature: Automatic Chinese to English translation

### Alternative Model Configurations
The system also supports the following model configurations (requires corresponding weight files):
- GroundingDINO_SwinB_cfg: Uses SwinB backbone (larger, more accurate but slower)

---

## 模型性能对比 / Model Performance Comparison

| 模型配置 / Model Config | 骨干网络 / Backbone | 精度 / Accuracy | 速度 / Speed | 内存占用 / Memory |
|------------------------|-------------------|-----------------|-------------|------------------|
| SwinT-OGC (当前/Current) | Swin Transformer-Tiny | 高 / High | 200ms | 较低 / Lower |
| SwinB (可选/Optional) | Swin Transformer-Base | 更高 / Higher | 300ms+ | 较高 / Higher |

## 更换模型说明 / Model Switching Instructions

如需更换为其他模型配置，请：
To switch to other model configurations:

1. 下载对应的预训练权重文件 / Download corresponding pretrained weights
2. 修改配置文件路径 / Modify configuration file path  
3. 更新模型加载代码 / Update model loading code
4. 重启服务器 / Restart server

更多详细信息请参考主README文件。
For more details, please refer to the main README files.
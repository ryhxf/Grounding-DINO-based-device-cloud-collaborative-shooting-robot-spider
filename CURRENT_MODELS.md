# Quick Model Reference / 快速模型参考

## 当前使用的模型 / Currently Used Models

### 🎯 主检测模型 / Primary Detection Model
**Grounding DINO with SwinT-OGC**
- 配置文件: `GroundingDINO_SwinT_OGC.py`
- 权重文件: `groundingdino_swint_ogc.pth`
- 性能: 200ms per image (640x480 on 4080 Super)

### 🔤 文本编码器 / Text Encoder  
**BERT-base-uncased**
- 路径: `local_models/bert-base-uncased/`
- 用途: 处理文本提示词

### 🌐 翻译模型 / Translation Model
**离线中英翻译 / Offline Chinese-English**
- 路径: `offline-zh-en-model/zh-en-model/`
- 功能: 中文→英文自动翻译

---

## 模型切换 / Model Switching

如需使用SwinB配置（更精确但更慢）:
To use SwinB configuration (more accurate but slower):

1. 下载SwinB权重文件 / Download SwinB weights
2. 修改配置为 `GroundingDINO_SwinB_cfg.py`
3. 重启服务器 / Restart server

详细信息请查看 [MODEL_INFO.md](MODEL_INFO.md)
For details, see [MODEL_INFO.md](MODEL_INFO.md)
**其他语言版本: [English](README.md), [中文](README_cn.md).**

代码都放在[release](https://github.com/ryhxf/Grounding-DINO-based-device-cloud-collaborative-shooting-robot-spider/releases/tag/v1.0)

📋 **当前使用的模型信息请查看：[MODEL_INFO.md](MODEL_INFO.md)**

# 基于多模态融合的端云协同军事射击机器人

![image](https://github.com/user-attachments/assets/90377063-cfa7-491c-842c-ccc0a36b71c6)

这篇参考文档写的比较随意，没技术报告里面那么多漂亮话，更多的实际开发调试中总结的经验，语言会比较朴素。
## 设计初衷
一开始只是想简单做个基于YOLO检测+PID控制的射击云台配合一辆小车。后面觉得吃技术老本还是太LOW了，想玩一玩最近比较火的多模态大模型，顺便再把老学长传承的六足机器人给用上，自此，便有了此项目。

一开始在考虑射击的时候，在思考能不能不仅仅就射人，而是我啥都能射，指哪打哪。考虑到应用场景和应用任务可以很复杂，要把各种各样的数据都考虑到，太麻烦了。因此想选择一些具有泛化性、零样本训练的模型。例如很火的Clip模型，不过之前在RDK-X5上也体验过clip模型，只能做到简单的分类，而并非目标检测，所以就继续实验一些视觉大模型。

在使用LLAMA Factory体验了一些视觉大模型后，例如QwenVL2.5，感觉输出的效率还是太慢了，而且每次都需要对大模型的回答进行解析处理，那样效率太低了。在不懈努力下，发现Grounding DINO模型，它是一种文本+图像的双模态模型。在输入文本提示词后，检测图像后直接输出检测框信息，单640*480图像在服务器4080super的单卡上最快可以跑到200ms，相比于其他大模型来说简直快到飞起。因此整个项目都基于该模型作为主体，包括后面的三级级联和端云协同方案。

至于为什么是六足机器人而不是四足机器人，原因有三：一方面是因为现在机器狗做的太火了，成本也比较高。另一方面是因为足数越少，姿态控制越难，机器狗的动作姿态啥的还需要调参，工作量太大。六足冗余度高，走路姿态随便控制就能保持平衡。最后是因为这个六足是老学长传承的，零成本（其实这才是主要原因）
## 功能介绍
整个机器人的功能需要实现自动跟踪射击、机器人移动、网页端远程操控。
前两个功能可以理解，但是为啥是网页端，主要是因为一开始是想做纯嵌入式端的语音控制，加个麦克风，连个讯飞大模型的事，后来考虑了一下应用场景，执行射击任务时应当是不能发出巨大声响的（虽然射击声音和舵机驱动声音已经够大了），至少不能大声密谋吧。所以就改做网页端了，相当于一个远程操控手能实时观测。
## 整体方案设计 

![image](https://github.com/user-attachments/assets/93eedb3b-d30a-4adf-834e-b43cb197d0f0)

实际设计参照上图，三端协同采用的是云服务器端、嵌入式端和网页端。其实更准确的来说，网页端应该算是云服务器端，但从使用者的角度来说，网页端还是应当独立出来比较直观一点。

服务器端用带NVIDIA显卡的电脑即可，主要是因为跑大模型都需要用CUDA加速，当然有些游戏笔记本自带独显的也可以作为服务器使用，而且还穿透内网还比较简单。不过我这里就直接用学校提供的服务器了，至于怎么穿透校园内网后面会说。

网页端就写了一个Web应用程序，当然代码是用AI辅助构建的，我就修改了一些共享参数显示，因为本人不是学计算机出身，是自动化本科，前后端对我来说算是外行技术。

嵌入式端用的开发板分别是树莓派4B、RDK-X5、GD32F303。六足机体用的是幻尔科技家的Spider，它上面自带了树莓派4B作为主控制器，把运动控制的任务分离出来了，就像是人体“小脑”一样。这倒是减轻了RDK-X5的负担。RDK-X5的任务就比较多了，更像是大脑，上通过HTTP网络与云服务器进行通信，实际用的是UDP协议。下还需要与树莓派和GD32完成控制机体，指挥射击等操作。跟踪射击部分用的是GD32单片机完成控制，与RDK-X5使用串口通信，为了接线方便，直接用了USB模拟串口，32这边就用了先进的Type-C接口，32的控制功能主要是通过高低电平控制枪完成射击，通过PWM波控制两个舵机，实现自动跟踪。

### Grounding DINO模型介绍
**当前使用模型：Grounding DINO with SwinT-OGC**

这个模型的内在核心原理，我就不仔细介绍了，毕竟在这个项目里只需要把它当成一个模块，做到怎么用，怎么调参就行了。

![image](https://github.com/user-attachments/assets/5861fbe8-f05f-4f18-9a93-9a7ccb89e6bd)

如上图，将文本关键词白色丝袜和对应图像输入Grounding DINO检测模型，最后直接会输出检测信息，包括检测框的大小、位置、置信度等参数。

**模型详细信息：**
- 模型配置：GroundingDINO_SwinT_OGC.py
- 预训练权重：groundingdino_swint_ogc.pth (~690MB)
- 骨干网络：Swin Transformer (SwinT)
- 检测速度：单640*480图像最快200ms (4080 Super GPU)
- 文本编码器：BERT-base-uncased

不过，该模型的输入文本关键词固定为英文，为了因此代码中还部署了自动检测中英文和离线翻译模型，使用时输入中文时，会自动翻译成英文输入模型。

### 端云协同机制下“大检测-小检测-跟踪”三级级联框架

![image](https://github.com/user-attachments/assets/433b6326-f3fa-4f63-a1ff-7b6e1e99810b)

其实一开始考虑采用云端检测+端侧跟踪的方案，例如Grounding DINO（云）+Deepsort跟踪（端）的方案，但实际上运行会出现很多问题，因为检测模型每次检测会有将近至少200ms以上的延迟，一次检测+一次跟踪就会导致跟踪帧率上限只能与检测帧率持平，只能做到堪堪3-5帧，这很明显是不够的，而且RDK-X5采用的BPU并没有针对Deepsort模型做优化适配，故采用CPU纯运行Deepsort的情况下，仅有0.5帧，甚至比检测帧率慢了将近10倍。因此换成了Grounding DINO（云）+Bytetrack（端）方案，把跟踪模型换成了无图像处理的轨迹跟踪Bytetrack，但实际也会出现时间滞后现象。根本原因在于想要同时适配大模型检测的低速和跟踪的高速，就必须加上一个“小检测”环节，即模板匹配算法，根据大模型的检测结果制作“模板”，再通过模板匹配完成快速检测，利用快速检测配合简单的Bytetrack跟踪算法完成最终的跟踪。在最终图像处理任务分配上采用“图像采集（端）+Grounding DINO（云）+模板匹配（云）+Bytetrack跟踪（云）+结果接收（端）”的架构，即“单次大模型检测+多次小检测+多次跟踪”的方法。

为啥模板匹配算法依旧在云端进行？
因为模板匹配算法也有CPU普通版本和GPU的Cuda加速版本，实际在RDK-X5上用CPU跑模板匹配，检测效果与Deepsort算法一样，都很慢，因此想要实现完美部署“Grounding DINO（云）+模板匹配（端）+Bytetrack跟踪（端）”，需要针对RDK-X5的BPU做加速算法适配，或者基于其他的AI开发板的NPU做加速算法适配。最简单的方法是换成NANO，直接调用NANO本身的GPU。因此，受限于当前硬件和本人技术力不足，采用了保守的纯云端处理方案，此方案的优点在于可以让帧率上限达到图像发送帧率，而不受限于大模型本身处理帧率。

### 自适应像素-角度数学映射模型
在使用该模型前，已经实验过PID方案。

![image](https://github.com/user-attachments/assets/d2da8d8a-a4bb-44f5-b801-9743bd5bd436)

如上图为枪眼一体安装方案，即摄像头跟随枪移动，PID的误差取图中右侧的计算方式，但经过测试效果很差。
如果为了追求PID控制的响应速度，就会出现响应快但抖动严重的问题，如果采用相对保守的PID参数，就会出现控制虽然很稳定，但是恢复速度非常慢。
深究原因，主要是因为舵机本身很难做到高精度的丝滑控制，其次就是跟踪射击会给摄像头带来大幅度的抖动，再加上摄像头是成本很低的640*480的USB免驱摄像头，也没有所谓的防抖和自动聚焦功能。因此就换成了枪眼分离的方案，也就是现在使用的数学映射模型方案。

![image](https://github.com/user-attachments/assets/a534e09d-54c4-4aa0-ac75-d5bb1b811a6d)

上图为安装的位置，使用的自适应像素-角度数学映射模型其实只是对舵机角度和像素点位置构建了数学映射关系，即当前需要射击的点位在图像中位置对应固定的舵机角度，在实际中当然不能使用穷举的方法进行数据记录，也需要提前设定几个参照点进行映射模型构建，具体使用了RBF插值模型，原理不进行详细赘述。

![image](https://github.com/user-attachments/assets/226d69b4-ca0a-4c85-aa32-7901387a4d16)

为了方便使用，写了一个本地的采集数据点的测试程序，程序为mult_client文件夹的coordinate_calibration.py

![image](https://github.com/user-attachments/assets/e8d8e98b-1c4d-434a-af4d-d6c0f86213b8)

操作方式很简单，通过WASD键可以控制云台上下左右移动，按F键可以进行射击，按数字123可以调节舵机的移动精度，按J进行保存，按ESC退出页面，如果不按J则不更新此次新标记的点。

![image](https://github.com/user-attachments/assets/bed7a70d-e4d6-4fb8-a7fb-80a061aa7f56)

在使用的过程中，鼠标直接点击枪头的红色激光头显示在图像中的位置即可进行标记，非常简单。标记成功后，进行随机移动，再次点击新的红色激光点位即可，当然也可以直接按F键进行模拟射击。

![image](https://github.com/user-attachments/assets/9a964859-b521-45fe-b1f3-7713d83047bc)

当按J保存后，会显示预标记数据点所在文件以及保存数量，文件保存当前执行指令的路径下，文件名为servo_calibration.json

![image](https://github.com/user-attachments/assets/6e013be6-6e3c-4b3f-8b94-f178a411cba1)

在实际射击使用中，系统就会读取servo_calibration.json的标记数据点完成模型构建与坐标映射。
### 机体控制逻辑
机体控制采用比较简单的预设动作组的控制方法，使用教程参考官方链接https://www.hiwonder.com.cn/course-detail/SpiderPi-Intelligent-Visual-Robot.html

![image](https://github.com/user-attachments/assets/f2317a7f-dc3b-4320-92ac-73b3db66f377)

此处为了方便，直接使用官方预设的动作组，预设动作为：前进、后退、左自转、右自转和高度控制。控制逻辑为：
当目标物体检测框在图像左侧时，机器人会主动完成左自转运动；
当目标物体检测框在图像右侧时，机器人会主动完成右自转运动；
当目标物体检测框在图像中面积过大时，机器人会主动完成后退运动；
当目标物体检测框在图像中面积过小时，机器人会主动完成前进运动；
当目标物体检测框在图像中位置过高时，机器人会主动抬高底盘；
当目标物体检测框在图像中位置过低时，机器人会主动降低底盘；
运动控制有利于射击跟踪控制，防止射击目标丢失。但两种控制在核心逻辑上并无交叉影响，在程序设计中，端侧的两种控制为独立的进程。
由于调试过程中有点BUG，现版本该部分功能先隐去。

### 网络设计
考虑到机器人在外执行任务时（自带网卡），需要和服务器构建通信网络，因此需要穿透内网，我们采用的是市面上较成熟的贝瑞蒲公英平台:
https://console.sdwan.oray.com/zh/main
该平台可以在不同设备之间构建局域网，实现端口映射和异地组网，注册账号后，有3个设备的免费额度，有2M的免费带宽可使用

![image](https://github.com/user-attachments/assets/efd6d519-fa96-4376-8ede-49fcd398c29b)

在网络成员中构建3个账号，分别对应三个设备，服务器端、嵌入式端、电脑操作端。使用前，需要在这三个设备上都安装贝瑞蒲公英客户端（注意不要下载服务器端），嵌入式端对应Linux的arm版本，电脑操作端对应Windows版本，服务器端对应Linux的X86版本。3个账号有对应的UID：XXXXXX：001/002/003，分别用于三台设备的账号登录。
还需要获取每个账号的虚拟IP（本人是172.16.X.XXX），在服务器端和嵌入式端对应的代码中需要修改为对应发送的网络IP地址。

![image](https://github.com/user-attachments/assets/a6bfa434-cf23-445a-99e5-9b45ee9e3f03)

电脑操作端主要是用户用于远程操控，只负责下发指令和观测状态，对带宽占用和实时性要求不高。
但是服务器端和嵌入式端需要进行大量的图像信息传输，需要保证实时性，因此需要特殊对待。

![image](https://github.com/user-attachments/assets/576544c5-7a9f-433d-8913-c5c0efa04b58)

如上图，为保证实时传输的低延迟，选用了速度更快、延迟更低的UDP网络传输协议，在传输过程中，加入图像压缩和解码，减少网络带宽压力。

### 网页界面设计
这是整个网页端界面设计

![image](https://github.com/user-attachments/assets/4684bc9f-472e-48f9-9822-f321c75432d1)

这是实时检测视频流

![image](https://github.com/user-attachments/assets/95ead5fc-310c-43fa-9b5a-c4fe9c23f5c5)

这是射击控制面板和控制面板，射击控制面板可以进行多目标射击和单目标射击（顺序全目标射击或指定目标），用于下达指令，控制面板用于设置检测的任意关键词，如人、红色灭火器、大树等等。

![image](https://github.com/user-attachments/assets/5d71a8b1-0c89-4dac-b8a7-9aa16a7a4d90)

还有实时帧率监控，可以观测最后跟踪的帧率和大模型的检测帧率与处理时间。
![image](https://github.com/user-attachments/assets/c54b5b3d-129a-4624-b698-a53a143398e1)


### 硬件设计
一开始构思的机械建模图

![image](https://github.com/user-attachments/assets/6410a4e1-d874-4ee0-bd53-d985709ca0f3)

GD32主控板开源在立创开源广场，网站如下：

[https://oshwhub.com/pursue.c/two-dimensional-shooting-ptz-mai]

![image](https://github.com/user-attachments/assets/719ae806-6c3e-4ee9-a7b1-3b2ed879cd37)

![image](https://github.com/user-attachments/assets/21255b7a-a224-4b91-865a-80ff97d9a317)

整体实物图

![image](https://github.com/user-attachments/assets/532cf04b-8060-4acd-949f-c5a87ac9d2bd)

## 实验效果

见B站视频
http

## 环境安装与源码
服务器端主要基于Grounding DINO工程进行开发，当时安装学习来源于别人做的B站教程

[https://www.bilibili.com/video/BV1dU1BYXEgj/?share_source=copy_web&vd_source=e6717e87e329b2362c9d525783220529]

和CSDN博客

[https://blog.csdn.net/weixin_44362044/article/details/136136728?ops_request_misc=elastic_search_misc&request_id=9ff952fe760454087ed4516a62fb9d0c&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-136136728-null-null.142^v102^pc_search_result_base9&utm_term=grounding%20dino&spm=1018.2226.3001.4187]

Grounding DINO源码工程

https://github.com/IDEA-Research/GroundingDINO

服务器端工程在conda虚拟环境中运行，安装Grounding DINO成功后，还需要安装一些其他的库，但是环境太多了，这里我给出一些比较必要的库，如果还有缺失麻烦自行补足，我把我的环境库放在server_requirement.txt里面了（用于备用对照参考）

```python==3.8.19```

```opencv-python==4.10.0.84```

```Flask==3.0.3```

```Flask-SocketIO==5.5.1```

```torch==2.0.1```

```torchvision==0.15.2```

```torchaudio==2.0.2```

客户端的RDK-X5基于Ubuntu22.04系统开发
以下是一些补充安装指令，可能不全，请麻烦自行补足，后续会慢慢补充。

```pip install opencv-python```

```pip install numpy```

```pip install pyserial```

```pip install pillow```

## 代码结构
**当前使用模型配置：Grounding DINO with SwinT-OGC**

服务器的工程文件是直接在Grounding DINO的工程基础上进行开发，因此主要详细展示增加的开发部分，其余部分有很多废案并未删除。

服务器端
```
GroundingDINO/                                    # GroundingDINO主项目根目录
├── groundingdino/                                # GroundingDINO核心算法模块（官方提供）
│   ├── __init__.py                              # 模块初始化文件
│   ├── version.py                               # 版本信息
│   ├── config/                                  # 模型配置文件目录
│   │   ├── __init__.py                          
│   │   ├── GroundingDINO_SwinT_OGC.py          # SwinT-OGC模型配置（当前使用）
│   │   └── GroundingDINO_SwinB_cfg.py          # SwinB模型配置（可选配置）
│   │   └── GroundingDINO_SwinB_cfg.py          # SwinB模型配置
│   ├── util/                                    # 工具函数模块
│   │   ├── __init__.py
│   │   ├── inference.py                        # 推理核心函数（load_model, predict, annotate）
│   │   ├── slconfig.py                          # 配置解析工具
│   │   ├── get_tokenlizer.py                    # 分词器获取
│   │   ├── box_ops.py                           # 边界框操作工具
│   │   ├── visualizer.py                        # 可视化工具
│   │   ├── utils.py                             # 通用工具函数
│   │   ├── logger.py                            # 日志工具
│   │   ├── time_counter.py                      # 时间计数器
│   │   ├── misc.py                              # 杂项工具
│   │   ├── slio.py                              # 文件I/O工具
│   │   └── vl_utils.py                          # 视觉-语言工具
│   ├── datasets/                                # 数据集处理模块
│   │   ├── __init__.py
│   │   ├── transforms.py                        # 图像变换处理
│   │   └── cocogrounding_eval.py                # COCO数据集评估
│   └── models/                                  # 模型定义模块
│       ├── __init__.py
│       ├── registry.py                          # 模型注册器
│       └── GroundingDINO/                       # GroundingDINO模型实现
│           ├── __init__.py
│           ├── groundingdino.py                 # 主模型定义
│           ├── bertwarper.py                    # BERT文本编码器
│           ├── ms_deform_attn.py                # 多尺度可变形注意力
│           ├── fuse_modules.py                  # 特征融合模块
│           ├── transformer.py                   # Transformer架构
│           ├── transformer_vanilla.py           # 标准Transformer
│           ├── utils.py                         # 模型工具函数
│           ├── backbone/                        # 骨干网络
│           │   ├── __init__.py
│           │   ├── backbone.py                  # 骨干网络基类
│           │   ├── swin_transformer.py          # Swin Transformer实现
│           │   └── position_encoding.py         # 位置编码
│           └── csrc/                            # C++/CUDA源码（编译优化）
│               ├── vision.cpp                   # 视觉处理C++代码
│               ├── cuda_version.cu              # CUDA版本检查
│               └── MsDeformAttn/                # 多尺度可变形注意力CUDA实现
│                   ├── ms_deform_attn.h
│                   ├── ms_deform_attn_cpu.h
│                   ├── ms_deform_attn_cpu.cpp
│                   ├── ms_deform_attn_cuda.h
│                   ├── ms_deform_attn_cuda.cu
│                   └── ms_deform_im2col_cuda.cuh
│
├── weights/                                      # 预训练模型权重目录
│   └── groundingdino_swint_ogc.pth              # SwinT-OGC预训练权重（约690MB）- 当前使用
│
├── local_models/                                 # 下载好的模型权重文件
│   └── bert-base-uncased/                       # BERT本地模型 - 当前使用
│       ├── config.json                          # BERT配置文件
│       ├── pytorch_model.bin                    # BERT模型权重
│       ├── tokenizer.json                       # 分词器配置
│       ├── tokenizer_config.json                # 分词器配置
│       ├── vocab.txt                            # 词汇表
│       └── special_tokens_map.json              # 特殊token映射
│
├── demo/                                        # 演示脚本目录（官方提供）
│   ├── inference_on_a_image.py                  # 单图像推理演示
│   ├── create_coco_dataset.py                   # COCO数据集创建
│   ├── test_ap_on_coco.py                       # COCO数据集AP测试
│   ├── image_editing_with_groundingdino_stablediffusion.ipynb  # 图像编辑演示
│   └── image_editing_with_groundingdino_gligen.ipynb           # GLIGEN图像编辑演示
│
│
├── server_web/                                  # Web服务器部分代码（本人增加部分）
│   ├── __init__.py
│   ├── web_server.py                           # 主Web服务器
│   ├── route_handlers.py                       # 路由处理器
│   ├── page_renderer.py                        # 页面渲染器
│   ├── image_utils.py                          # 图像处理工具
│   └── font_handler.py                         # 字体处理器
│
│
├── offline-zh-en-model/                        # 离线中英翻译模型（本人增加部分）
│   └── zh-en-model/                            # 中英翻译模型文件 - 当前使用
│       ├── config.json                         # 模型配置
│       ├── generation_config.json             # 生成配置
│       ├── tokenizer_config.json              # 分词器配置
│       ├── vocab.json                          # 词汇表
│       ├── source.spm                          # 源语言模型
│       ├── target.spm                          # 目标语言模型
│       └── pre_coordinate_calibration.py      # 预标定脚本
│
│
├── modules/                                     # 🔥 核心模块目录（本人增加部分）
│   ├── __init__.py                             # 模块初始化文件
│   ├── dino_module.py                          # GroundingDINO模型核心实现模块
│   ├── image_receiver.py                       # 图像接收和处理模块  
│   ├── web_module.py                           # Web界面服务模块（主要在server_web中体现）
│   ├── template_tracker.py                     # 模板跟踪算法模块
│   ├── trans.py                                # 翻译模块
│   ├── download_model.py                       # 下载模型代码
│   └── 修改贴士.md                             # 开发修改提示文档
│
└── mult/                                        # 🔥 服务器端核心代码目录（多进程架构）（本人增加部分）
    ├── __init__.py                              # 模块初始化文件
    ├── init.py                                  # 初始化脚本
    ├── README.md                                # 多进程系统说明文档
    ├── run_server.py                            # 🚀 服务器启动脚本（主入口）
    ├── server_multiprocess.py                  # 🔥 核心多进程服务器实现
    ├── image_receiver_multiprocess.py          # 🔥 图像接收多进程处理器
    ├── offline_bert_setup.py                   # 离线BERT模型配置脚本
    └── cache/                                   # 缓存配置目录

```

嵌入式端（RDK-X5）
```
├── mult_client/                                 # 多进程客户端系统（本人增加部分）
│   ├── client_multiprocess.py                  # 🔥 核心多进程客户端（图像采集、UDP发送、跟踪接收）
│   ├── run_client.py                           # 客户端启动脚本
│   ├── coordinate_mapper.py                    # 坐标映射控制器（屏幕坐标转换）
│   ├── coordinate_calibration.py              # 坐标标定程序
│   ├── pre_coordinate_calibration.py          # 预标定程序
│   ├── pid_control.py                          # PID控制器和串口通信
│   ├── serial_sender_process.py               # 串口发送进程
│   ├── serial_test.py                          # 串口测试程序
│   ├── serial_test_multiprocess.py            # 多进程串口测试
│   ├── test_serial_sender.py                  # 串口发送测试
│   ├── usb_detector.py                        # USB设备检测工具
│   ├── servo_calibration.json                 # 标定数据文件
```

嵌入式端（GD32）（OLED模块未启用）
```
GD32/                                          # 主项目根目录
├── Core/                                      # 核心应用代码目录
│   ├── Inc/                                   # 头文件目录
│   │   ├── gd32f30x_it.h                     # 中断服务函数声明
│   │   ├── gd32f30x_libopt.h                 # GD32标准库配置选项
│   │   ├── gpio.h                            # GPIO引脚配置声明
│   │   ├── main.h                            # 主程序头文件，包含主要依赖
│   │   ├── OLED.h                            # OLED显示屏驱动头文件
│   │   ├── proj_config.h                     # 项目配置参数
│   │   ├── Serial.h                          # 串口通信头文件
│   │   ├── steer1.h                          # 舵机控制头文件（双舵机系统）
│   │   ├── systick.h                         # 系统滴答定时器头文件
│   │   ├── tim.h                             # 定时器配置头文件
│   │   └── usart.h                           # USART串口驱动头文件
│   └── Src/                                  # 源文件目录
│       ├── gd32f30x_it.c                     # 中断服务函数实现
│       ├── gpio.c                            # GPIO引脚初始化和控制
│       ├── main.c                            # 主程序入口（包含舵机控制逻辑）
│       ├── Serial.c                          # 串口通信实现
│       ├── steer1.c                          # 舵机PWM控制实现
│       ├── system_gd32f30x.c                 # GD32系统初始化
│       ├── systick.c                         # 系统滴答定时器实现
│       ├── tim.c                             # 定时器配置和PWM生成
│       └── usart.c                           # USART串口驱动实现
│
├── Drivers/                                   # 驱动程序目录
└── MDK-ARM/                                  # Keil MDK开发环境配置
```
## 代码使用指令
**服务器端运行**
跳转到工程文件夹下（这里我用的自己的路径，请自行修改）

```cd /home/ubuntu/beifen/yyx_code/big_model/GroundingDINO```

执行服务器端代码（指定网页端的映射端口号，因为需要经常更换，有时候有些端口生成网页端口不好使）

```python python mult/run_server.py --web-port 4567 ```

![image](https://github.com/user-attachments/assets/b4eb3094-2025-477b-8bb7-358763324651)

![image](https://github.com/user-attachments/assets/f6f34839-13d5-4c4b-b40e-8a39fa95b38c)

出现上图结果说明启动成功了，接下来启动客户端即可

可配置参数
--image-port      # 图像接收端口（嵌入式端发送给服务器端的网络端口号）
--tracking-port   # 客户端跟踪结果接收端口（服务器端发送给嵌入式端跟踪结果的网络端口号）
--web-port        # Web网页端口号
--device          # 指定服务器使用的显卡号，示例'cuda:3'
--detect-time     # Grounding DINO模型自动检测间隔时间（秒）

需要修改的参数
server_multiprocess.py代码137、478行左右把172.16.3.220修改成自己的嵌入式端自己的虚拟网络IP（注意：这是服务器端，所以是写的是要发送的嵌入式端IP，不是自己的IP）

**嵌入式端（RDK-X5）运行代码**

```python run_client.py```

client_multiprocess.py代码
997行的心跳包机制中改成客户端（嵌入式端）的IP号（也就是自己的IP号）
1327行main函数部分的参数修改成自己需要的即可
--server-ip # 服务器IP地址
--image-port # 图像发送端口
--camera-id # 摄像头设备ID号
--timeout # 连接超时时间
--image-quality # 发送图像质量
--tracking-port # 跟踪结果接收端口（本地监听）

注意：必须要修改1330行服务器端的IP号（注意：这是嵌入式端，所以是写的是要发送的服务器端IP，不是自己的IP）

客户端的跟踪结果接收端口如果想自行更换，需要修改部分较多，在main函数的1336行，81行，451行的tracking_port参数进行修改

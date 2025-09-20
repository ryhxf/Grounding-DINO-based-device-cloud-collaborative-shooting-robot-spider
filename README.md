**Read this in other languages: [English](README.md), [中文](README_cn.md).**

The code is all placed here [release](https://github.com/ryhxf/Grounding-DINO-based-device-cloud-collaborative-shooting-robot-spider/releases/tag/v1.0)

📋 **For current model information, see: [MODEL_INFO.md](MODEL_INFO.md)**

# End-cloud collaborative military shooting robot based on multi-modal fusion

![image](https://github.com/user-attachments/assets/90377063-cfa7-491c-842c-ccc0a36b71c6)

This reference document is written in a casual way, without as many beautiful words as technical reports, and more experience summarized in actual development and debugging. The language will be relatively simple.
## Design intent
Initially, I simply wanted to create a shooting gimbal system combining YOLO detection and PID control with a vehicle. However, I realized relying on technical expertise was too limited and decided to explore the trending multi-modal large models while incorporating the six-legged robot developed by my senior mentor. This marked the beginning of this project. When considering shooting capabilities, I pondered whether the system could target multiple targets rather than just humans. Given the complexity of application scenarios and tasks requiring diverse data, I prioritized generalizable models like the popular Clip model. However, my previous experience with the RDK-X5 only allowed basic classification, not object detection, so I experimented with other visual large models. After testing visual models through LLAMA Factory (e.g., QwenVL2.5), I found their output efficiency too slow and required processing of large model responses, significantly reducing efficiency. Through persistent efforts, Grounding DINO emerged as the breakthrough—a text-image dual-modal model. By inputting text prompts and detecting images, it directly outputs bounding box information. On a single 4080 Super GPU, processing a 640x480 image takes just 200ms—far faster than other large models. The entire project now centers around this model, including its three-stage cascade architecture and end-cloud collaboration solution. There are three key reasons why we opted for six-legged robots over four-legged ones. First, robotic dogs have become highly popular and expensive to maintain. Second, fewer legs make posture control more challenging – adjusting parameters for robotic dog movements requires significant effort. Sixth-legged robots, with their high redundancy, can maintain balance simply by controlling walking postures. Lastly, this six-legged design carries forward the legacy of our predecessors, being cost-free (though this is actually the main reason).
## Function Introduction
The robot's core functionalities include automatic tracking and shooting, mobile navigation, and web-based remote control. While the first two features are straightforward, the web interface was chosen because the initial plan focused on pure embedded voice control with a microphone and iFLYTEK's large model. However, practical considerations revealed that shooting operations must remain low-volume (despite existing loud noises from the shooter and servo motors), especially when conducting covert maneuvers. This necessitated transitioning to web-based control, enabling real-time monitoring by remote operators.
## Overall scheme design 

![image](https://github.com/user-attachments/assets/93eedb3b-d30a-4adf-834e-b43cb197d0f0)

The actual design reference is shown in the diagram, where the three-end collaboration system integrates cloud server, embedded device, and web interface. To be precise, the web interface should technically belong to the cloud server side, but from a user perspective, it's more intuitive to maintain its independence. The server side requires a computer with NVIDIA graphics cards, as large model training necessitates CUDA acceleration. While gaming laptops with integrated graphics can also serve as servers, they're easier to penetrate the campus network. For this project, I directly utilized school-provided servers, though details about internal network penetration will be discussed later. The web interface features an AI-assisted-built Web application, with shared parameters modified for display purposes. As an automation engineering graduate rather than a computer science background, I consider both front-end and back-end technologies as peripheral expertise. The embedded systems utilize Raspberry Pi 4B, RDK-X5, and GD32F303 development boards. The six-legged robot employs Phantom Robotics 'Spider module, which integrates Raspberry Pi 4B as its main controller to separate motion control tasks—similar to a human cerebellum. This design reduces RDK-X5's workload, while its primary function resembles a brain node: communicating with cloud servers via HTTP (though using UDP in practice) and coordinating operations like shooting commands with Raspberry Pi and GD32. The tracking and shooting functions are controlled by the GD32 microcontroller, which communicates with the RDK-X5 via a serial port. For easier wiring, a USB analog serial port was directly used, while the 32-bit microcontroller employs an advanced Type-C interface. The control mechanism primarily uses high-low voltage signals to operate the gun for shooting, and PWM waves to control two servos for automatic tracking.
### Introduction to the DINO Grounding Model
I won't go into the core principles of this model, since in this project you just need to think of it as a module and figure out how to use it and how to tune the parameters.

![image](https://github.com/user-attachments/assets/5861fbe8-f05f-4f18-9a93-9a7ccb89e6bd)

As shown in the figure above, when text keywords such as "white stockings" and corresponding images are input into the Grounding DINO detection model, it directly outputs detection results including parameters like bounding box size, position, and confidence level. However, since the model's input text keywords are fixed to English, the code also incorporates automatic detection of Chinese-English translation and offline translation models. When Chinese input is used during operation, it will be automatically translated into English for model processing.

### The three-level cascading framework of "large detection-small detection-tracking" under the end-cloud collaboration mechanism

![image](https://github.com/user-attachments/assets/433b6326-f3fa-4f63-a1ff-7b6e1e99810b)

Initially, we considered implementing a cloud-based detection system combined with edge-side tracking, such as Grounding DINO (cloud) + Deepsort tracking (edge). However, this approach encountered significant operational challenges. The detection model introduced at least 200ms latency per frame, resulting in tracking frame rates being limited to 3-5 frames per second – barely meeting basic requirements. Moreover, the BPU architecture in RDK-X5 lacked optimization for Deepsort models. When running Deepsort exclusively on CPU, the system achieved only 0.5 frames per second, nearly 10 times slower than detection performance. Consequently, we switched to the Grounding DINO (cloud) + ByteTrack (edge) solution, replacing the tracking model with ByteTrack's image-free trajectory tracking algorithm. This transition still resulted in time lag issues. The fundamental challenge lies in balancing low-speed detection and high-speed tracking for large models: To achieve both, we implemented a "small detection" phase using template matching algorithms. These algorithms generate templates based on detection results, enabling rapid initial detection. Subsequent tracking is then executed through simple ByteTrack algorithms, effectively combining fast detection with efficient edge processing. The architecture for final image processing task allocation adopts the "Image Capture (end) + Grounding DINO (cloud) + Template Matching (cloud) + ByteTrack tracking (cloud) + Result Reception (end)" framework, implementing a method of "single large model detection + multiple small detections + repeated tracking". Why is template matching still performed in the cloud? This is because template matching algorithms exist in both CPU-based standard versions and GPU-accelerated Cuda versions. In practice, running template matching on RDK-X5 with CPUs yields detection performance comparable to Deepsort, which remains slow. To achieve perfect deployment of "Grounding DINO (cloud) + template matching (end) + ByteTrack tracking (end)", accelerated algorithm adaptations are required either for RDK-X5's BPU or other AI development boards' NPU. The simplest solution would be switching to NANO, directly utilizing its built-in GPU. However, constrained by current hardware limitations and technical capabilities, we adopted a conservative pure cloud processing approach. This solution's advantage lies in maintaining frame rate ceilings at the image transmission rate, free from restrictions imposed by large model processing speeds.

### Adaptive pixel-angle mathematical mapping model
The PID scheme has been tested before using this model.

![image](https://github.com/user-attachments/assets/d2da8d8a-a4bb-44f5-b801-9743bd5bd436)

The above figure shows a gun-eye integrated installation solution, that is, the camera moves with the gun, and the PID error is calculated using the calculation method on the right side of the figure, but the effect is very poor after testing. If the response speed of PID control is pursued, there will be a problem of fast response but severe jitter. If relatively conservative PID parameters are used, the control will be very stable but the recovery speed will be very slow. The main reason is that it is difficult for the servo itself to achieve high-precision smooth control. Secondly, tracking shooting will cause large jitters to the camera. In addition, the camera is a very low-cost 640*480 USB driver-free camera, and there is no so-called anti-shake and auto-focus function. Therefore, the gun-eye separation solution was replaced, which is the mathematical mapping model solution currently used.

![image](https://github.com/user-attachments/assets/a534e09d-54c4-4aa0-ac75-d5bb1b811a6d)

The picture above shows the installation position. The pixel-angle mathematical mapping model used actually only constructs a mapping mathematical relationship between the servo angle and the pixel position, that is, the position of the point currently to be photographed in the image corresponds to a fixed servo angle. In practice, of course, the exhaustive method cannot be used for data recording. It is also necessary to set several reference points in advance to construct the mapping model. The RBF interpolation model is used specifically, and the principle will not be repeated here.

![image](https://github.com/user-attachments/assets/226d69b4-ca0a-4c85-aa32-7901387a4d16)

For ease of use, a test program for local data collection is written. The program is coefficient_calibration.py in the mult_client folder.

![image](https://github.com/user-attachments/assets/e8d8e98b-1c4d-434a-af4d-d6c0f86213b8)

The operation is very simple. Use the WASD keys to control the gimbal to move up, down, left, and right. Press the F key to shoot. Press the numbers 123 to adjust the movement accuracy of the servo. Press J to save. Press ESC to exit the page. If you do not press J, the marked point will not be updated.

![image](https://github.com/user-attachments/assets/bed7a70d-e4d6-4fb8-a7fb-80a061aa7f56)

During use, you can mark the position of the red laser head on the gun head directly by clicking the mouse on the position displayed in the image. It is very simple. After marking successfully, move randomly and click the new red laser point again. Of course, you can also directly press the F key to simulate shooting.

![image](https://github.com/user-attachments/assets/9a964859-b521-45fe-b1f3-7713d83047bc)

After pressing J to save, the file where the pre-marked data points are located and the number of saved files will be displayed. The file is saved in the path of the current execution command and the file name is servo_calibration.json

![image](https://github.com/user-attachments/assets/6e013be6-6e3c-4b3f-8b94-f178a411cba1)

In actual use, the system can read the marked data points of servo_calibration.json to complete model construction and coordinate mapping.
### Using control logic
At the same time, the control method of the relatively simple default action group is adopted. For the tutorial, please refer to the official link https://www.hiwonder.com.cn/course-detail/SpiderPi-Intelligent-Visual-Robot.html

![image](https://github.com/user-attachments/assets/f2317a7f-dc3b-4320-92ac-73b3db66f377)

For convenience, we directly use the official preset action group here. The preset actions are: forward, backward, left rotation, right rotation and height control. The control logic is: when the target object detection frame is on the right side of the image, the robot will actively complete the left rotation movement; when the target object detection frame is on the right side of the image, the robot will actively complete the right rotation movement; when the target object detection frame is too large in the image, the robot will actively complete the rear rotation movement; when the target object detection frame is too small in the image, the robot will actively complete the rear rotation movement; when the target object detection frame is too high in the image, the robot will actively start high molding; when the target object detection frame is too low in the image, the robot will actively reduce molding; motion control is conducive to tracking control to prevent the shooting target from being lost. However, the two controls have no cross-effect in logic. In program design, the two controls on the end side are the core process. Due to some BUGs in the debugging process, this part of the function is hidden in the current version.

### Network Design
Considering that the robot needs to build a communication network with the server when performing tasks externally (with its own bandwidth), we use the more mature Berry Dandelion platform: https://console.sdwan.oray.com/zh/main. This platform can deploy routers between different devices, requiring port mapping and remote networking. After registering an account, there is a free mode for 3 devices and 2M free bandwidth available.

![image](https://github.com/user-attachments/assets/efd6d519-fa96-4376-8ede-49fcd398c29b)

Build 3 accounts among the network members, one for each device, the server, the embedded, and the computer. Before use, you need to install the Berry Dandelion client on all three devices (be careful not to download the server). The embedded corresponds to the arm version of Linux, the computer corresponds to the Windows version, and the server corresponds to the X86 version of Linux. The three accounts have corresponding UIDs: XXXXXX: 001/002/003, corresponding to the login accounts of the three devices. You also need to modify the virtual IP for each account (mine is 172.16.X.XXX), and the corresponding network IP address needs to be sent in the corresponding code on the server and embedded sides.

![image](https://github.com/user-attachments/assets/a6bfa434-cf23-445a-99e5-9b45ee9e3f03)

The computer operation end is mainly controlled by the user remotely, and is only responsible for issuing instructions and starting the status, and does not require high bandwidth usage and real-time performance. However, the server and embedded end need to transmit a large amount of image information, and real-time performance must be guaranteed, which requires special treatment.

![image](https://github.com/user-attachments/assets/63f00478-23dc-4d90-b9ee-f81494b7e0dc)

As shown in the figure above, in order to ensure low latency in real-time transmission, the UDP network transmission protocol with faster speed and higher latency is adopted. During the transmission process, image compression and decoding are added to reduce the pressure on network bandwidth.

### Web interface design
This is the entire web interface design：

![image](https://github.com/user-attachments/assets/4684bc9f-472e-48f9-9822-f321c75432d1)

This is the real-time detection video stream

![image](https://github.com/user-attachments/assets/95ead5fc-310c-43fa-9b5a-c4fe9c23f5c5)

This is the shooting control panel and control panel. The shooting control panel can perform multi-target shooting and single-target shooting (sequential all-target shooting or designated targets). It is used to issue instructions. The control panel is used to set the same keywords for detection, such as people, red fire extinguishers, big trees, etc.

![image](https://github.com/user-attachments/assets/5d71a8b1-0c89-4dac-b8a7-9aa16a7a4d90)

There is also real-time frame rate monitoring, which can start the last tracked frame rate and the detection frame rate and processing time of large models.

![image](https://github.com/user-attachments/assets/c54b5b3d-129a-4624-b698-a53a143398e1)


### Hardware Design
Start building a mechanical modeling diagram

![image](https://github.com/user-attachments/assets/6410a4e1-d874-4ee0-bd53-d985709ca0f3)

The GD32 main control board is located in LiChuang Open Source Plaza, the website is as follows:
[https://oshwhub.com/pursue.c/two-dimensional-shooting-ptz-mai]

![image](https://github.com/user-attachments/assets/719ae806-6c3e-4ee9-a7b1-3b2ed879cd37)

![image](https://github.com/user-attachments/assets/21255b7a-a224-4b91-865a-80ff97d9a317)

Overall physical picture

![image](https://github.com/user-attachments/assets/532cf04b-8060-4acd-949f-c5a87ac9d2bd)

## Experimental results

See the video at Station B
http

## Environment installation and source code
The server side is mainly developed based on the Grounding DINO project. At that time, the installation and learning came from the B station tutorial made by others.

[https://www.bilibili.com/video/BV1dU1BYXEgj/?share_source=copy_web&vd_source=e6717e87e329b2362c9d525783220529]

and CSDN blog

[https://blog.csdn.net/weixin_44362044/article/details/136136728?ops_request_misc=elastic_search_misc&request_id=9ff952fe760454087ed4516a62fb9d0c&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-136136728-null-null.142^v102^pc_search_result_base9&utm_term=grounding%20dino&spm=1018.2226.3001.4187]

Grounding DINO Source code engineering

https://github.com/IDEA-Research/GroundingDINO

The server-side project runs in the conda virtual environment. After successfully installing Grounding DINO, you need to install some other libraries, but there are too many environments. Here I give some of the more necessary libraries. If there are any missing, please fill them in yourself. I put my environment library in server_requirement.txt (for backup reference)

```python==3.8.19```

```opencv-python==4.10.0.84```

```Flask==3.0.3```

```Flask-SocketIO==5.5.1```

```torch==2.0.1```

```torchvision==0.15.2```

```torchaudio==2.0.2```

The client's RDK-X5 is developed based on the Ubuntu 22.04 system.
The following are some additional installation instructions, which may not be complete. Please supplement them yourself. They will be supplemented later.

```pip install opencv-python```

```pip install numpy```

```pip install pyserial```

```pip install pillow```

## Code structure
**Current Model Configuration: Grounding DINO with SwinT-OGC**

The server project files are developed directly based on the Grounding DINO project, so the added development parts are mainly shown in detail, and there are many abandoned projects in the rest that have not been deleted.

Server Side
```
GroundingDINO/                                    # GroundingDINO main project root directory
├── groundingdino/                                # GroundingDINO core algorithm module (official)
│   ├── __init__.py                              # Module initialization file
│   ├── version.py                               # Version Information
│   ├── config/                                  # Model configuration file directory
│   │   ├── __init__.py                          
│   │   ├── GroundingDINO_SwinT_OGC.py          # SwinT-OGC model configuration (CURRENTLY USED)
│   │   └── GroundingDINO_SwinB_cfg.py          # SwinB model configuration (alternative)
│   ├── util/                                    # Tool function module
│   │   ├── __init__.py
│   │   ├── inference.py                        # Inference core functions (load_model, predict, annotate)
│   │   ├── slconfig.py                          # Configuration parsing tool
│   │   ├── get_tokenlizer.py                    # Get the word segmenter
│   │   ├── box_ops.py                           # Bounding Box Operation Tools
│   │   ├── visualizer.py                        # Visualization Tools
│   │   ├── utils.py                             # General utility functions
│   │   ├── logger.py                            # Logging Tools
│   │   ├── time_counter.py                      #Time counter
│   │   ├── misc.py                              # Miscellaneous Tools
│   │   ├── slio.py                              # File I/O Tools
│   │   └── vl_utils.py                          # Visual-linguistic tools
│   ├── datasets/                                # Dataset processing module
│   │   ├── __init__.py
│   │   ├── transforms.py                        # Image transformation processing
│   │   └── cocogrounding_eval.py                # COCO dataset evaluation
│   └── models/                                  # Model definition module
│       ├── __init__.py
│       ├── registry.py                          # Model Registry
│       └── GroundingDINO/                       # GroundingDINO model implementation
│           ├── __init__.py
│           ├── groundingdino.py                 # Main model definition
│           ├── bertwarper.py                    # BERT Text Encoder
│           ├── ms_deform_attn.py                # Multi-scale deformable attention
│           ├── fuse_modules.py                  # Feature fusion module
│           ├── transformer.py                   # Transformer Architecture
│           ├── transformer_vanilla.py           # Standard Transformer
│           ├── utils.py                         # Model tool functions
│           ├── backbone/                        # Backbone network
│           │   ├── __init__.py
│           │   ├── backbone.py                  # Backbone network base class
│           │   ├── swin_transformer.py          # Swin Transformer Implementation
│           │   └── position_encoding.py         # Positional encoding
│           └── csrc/                            # C++/CUDA source code (compilation optimization)
│               ├── vision.cpp                   # Visual Processing C++ Code
│               ├── cuda_version.cu              # CUDA version check
│               └── MsDeformAttn/                # Multi-scale Deformable Attention CUDA Implementation
│                   ├── ms_deform_attn.h
│                   ├── ms_deform_attn_cpu.h
│                   ├── ms_deform_attn_cpu.cpp
│                   ├── ms_deform_attn_cuda.h
│                   ├── ms_deform_attn_cuda.cu
│                   └── ms_deform_im2col_cuda.cuh
│
├── weights/                                      # Pre-trained model weights directory
│   └── groundingdino_swint_ogc.pth              # SwinT-OGC pre-trained weights (about 690MB) - CURRENTLY USED
│
├── local_models/                                 # Model weight files
│   └── bert-base-uncased/                        # BERT local model - CURRENTLY USED
│       ├── config.json                           # BERT configuration file
│       ├── pytorch_model.bin                     # BERT model weights
│       ├── tokenizer.json                        # Tokenizer configuration
│       ├── tokenizer_config.json                 # Tokenizer configuration
│       ├── vocab.txt                             # Vocabulary
│       └── special_tokens_map.json               # Special token mapping
│
├── demo/                                         # Demo scripts (official)
│   ├── inference_on_a_image.py                   # Inference on a single image
│   ├── create_coco_dataset.py                    # Create COCO dataset
│   ├── test_ap_on_coco.py                        # AP testing on COCO dataset
│   ├── image_editing_with_groundingdino_stablediffusion.ipynb  # Image editing demo with StableDiffusion
│   └── image_editing_with_groundingdino_gligen.ipynb           # GLIGEN image editing demo
│
│
├── server_web/                                   # Web server code (custom added)
│   ├── __init__.py
│   ├── web_server.py                             # Main web server
│   ├── route_handlers.py                         # Route handler
│   ├── page_renderer.py                          # Page renderer
│   ├── image_utils.py                            # Image processing utilities
│   └── font_handler.py                           # Font handler
│
│
├── offline-zh-en-model/                          # Offline Chinese-English translation model (custom added)
│   └── zh-en-model/                              # Translation model files - CURRENTLY USED
│       ├── config.json                           # Model configuration
│       ├── generation_config.json                # Generation configuration
│       ├── tokenizer_config.json                 # Tokenizer configuration
│       ├── vocab.json                            # Vocabulary
│       ├── source.spm                            # Source language model
│       ├── target.spm                            # Target language model
│       └── pre_coordinate_calibration.py         # Pre-calibration script
│
│
├── modules/                                      # Core modules (custom added)
│   ├── __init__.py                               # Module initializer
│   ├── dino_module.py                            # GroundingDINO core implementation
│   ├── image_receiver.py                         # Image reception and processing
│   ├── web_module.py                             # Web interface service (mainly used in server_web)
│   ├── template_tracker.py                       # Template tracking algorithm
│   ├── trans.py                                  # Translation module
│   ├── download_model.py                         # Model download script
│   └── 修改贴士.md                                # Tips for modifying code (Chinese documentation)
│
└── mult/                                         # Server-side core (multiprocessing architecture) (custom added)
    ├── __init__.py                                # Module initializer
    ├── init.py                                    # Initialization script
    ├── README.md                                  # Multiprocessing system documentation
    ├── run_server.py                              # Server startup script (main entry point)
    ├── server_multiprocess.py                     # Multiprocessing server implementation
    ├── image_receiver_multiprocess.py             # Image receiver multiprocessing handler
    ├── offline_bert_setup.py                      # Offline BERT model setup
    └── cache/                                     # Cache configuration

```

Embedded Side (RDK-X5)
```
├── mult_client/                                  # Multiprocessing client system (custom added)
│   ├── client_multiprocess.py                    # Core multiprocessing client (image capture, UDP send, tracking receive)
│   ├── run_client.py                             # Client startup script
│   ├── coordinate_mapper.py                      # Coordinate mapping controller (screen coordinate conversion)
│   ├── coordinate_calibration.py                 # Coordinate calibration
│   ├── pre_coordinate_calibration.py             # Pre-calibration
│   ├── pid_control.py                            # PID controller and serial communication
│   ├── serial_sender_process.py                  # Serial sending process
│   ├── serial_test.py                            # Serial testing script
│   ├── serial_test_multiprocess.py               # Multiprocess serial testing
│   ├── test_serial_sender.py                     # Serial sender test
│   ├── usb_detector.py                           # USB device detector
│   ├── servo_calibration.json                    # Calibration data

```

Embedded Side (GD32) (OLED module not used)
```
GD32/                                             # Main project root directory
├── Core/                                         # Core application code
│   ├── Inc/                                      # Header files
│   │   ├── gd32f30x_it.h                         # Interrupt service declarations
│   │   ├── gd32f30x_libopt.h                     # GD32 standard library config options
│   │   ├── gpio.h                                # GPIO configuration
│   │   ├── main.h                                # Main program header
│   │   ├── OLED.h                                # OLED display driver header
│   │   ├── proj_config.h                         # Project configuration parameters
│   │   ├── Serial.h                              # Serial communication
│   │   ├── steer1.h                              # Servo control header (dual servo)
│   │   ├── systick.h                             # System tick timer
│   │   ├── tim.h                                 # Timer configuration
│   │   └── usart.h                               # USART driver
│   └── Src/                                      # Source files
│       ├── gd32f30x_it.c                         # Interrupt service implementation
│       ├── gpio.c                                # GPIO initialization/control
│       ├── main.c                                # Main program entry (servo control logic)
│       ├── Serial.c                              # Serial communication
│       ├── steer1.c                              # Servo PWM control
│       ├── system_gd32f30x.c                     # GD32 system initialization
│       ├── systick.c                             # System tick implementation
│       ├── tim.c                                 # Timer and PWM
│       └── usart.c                               # USART driver
│
├── Drivers/                                      # Drivers
└── MDK-ARM/                                      # Keil MDK project files
```
## Command Usage Instructions
**Run on Server Side**
Navigate to the project folder (replace the path with your own):

```cd /home/ubuntu/beifen/yyx_code/big_model/GroundingDINO```

Run the server code and specify the web port (this is often changed due to port availability issues):

```python python mult/run_server.py --web-port 4567 ```

![image](https://github.com/user-attachments/assets/b4eb3094-2025-477b-8bb7-358763324651)

![image](https://github.com/user-attachments/assets/f6f34839-13d5-4c4b-b40e-8a39fa95b38c)

Once the output shown above appears, it indicates that the server has started successfully. Next, start the client.

Configurable Parameters
--image-port      # Port to receive images (the network port to which the embedded side sends images to the server)
--tracking-port   # Port to receive tracking results on the client side (server sends results to the embedded device through this port)
--web-port        # Web interface port number
--device          # Specify which GPU the server should use, e.g.,'cuda:3'
--detect-time     # Auto-detection interval for Grounding DINO (in seconds)

Parameters to Modify
In the server_multiprocess.py file, around lines 137 and 478, replace 172.16.3.220 with the virtual IP address of your embedded device.
**Running Code on the Embedded Side (RDK-X5)**

```python run_client.py```

In the client_multiprocess.py file:
At line 997, update the IP in the heartbeat mechanism to the client’s own IP address (i.e., the IP of the embedded device).
At line 1327, modify the parameters in the main function as needed:
--server-ip # IP address of the server
--image-port # Port for sending images
--camera-id # Camera device ID
--timeout # Connection timeout
--image-quality # Quality of the sent image
--tracking-port # Port to receive tracking results (local listener)

Notice： Important: At line 1330, you must change the server IP to the actual server IP address.
This is on the embedded side, so the IP should be the destination server IP, not the client’s own IP.
If you want to change the tracking result receiving port for the client, you'll need to update multiple places:
Line 1336 in the main function
Line 81 and line 451, where tracking_port is used

#  Robust Real-Time Stereo Matching via Context Encoded Iterative Refinement
![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/Context-Stereo/img/f5.jpg)


## 🌟 Environment

The hardware and software environments for training and testing are **NVIDIA RTX 3080 GPU**, Intel i9-12900k (12th gen), 32 GB memory, Ubuntu 22.04 LTS, Python 3.8, PyTorch 2.0.0, CUDA 11.8.

All experiments in the paper including comparisons, ablation experiments in Table 1 and runtime tests are using the above environment.

Pretraining on Scene Flow costs 11.2 h, finetuning on KITTI costs 2.6 h.

## 🏆 KITTI 2012 benchmark
**[Context-Stereo-I](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=all&result=29ba705148fd0bccf2f183180f2ca3d543778392)** Rank #93,

**[Context-Stereo](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=all&result=5e3ff6f4936e065626cf8ebb657bd89f9d1c98d0)** Rank #113.

## 🏆 KITTI 2015 benchmark
**[Context-Stereo-I](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=633a7405b3b2329c494f33ab9c2a954f801ddada)** Rank #95

**[Context-Stereo](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=19fafc7a0b041ccf935def0c20161f5446976e5f)** Rank #155.



## 📊 Table 1. Performance of ECF and AGS in existing methods on Scene Flow, KITTI 2015 benchmark, Middlebury and ETH3D.


| Method | Cost Volume | Cost Aggregation | Hourglass Number | Contextual Aggregation | Scene EPE (px) | KT 15 D1-all (%) | MID (Zero-shot) | ETH3D (Zero-shot) | Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fast-ACV-base0 | ACV | hourglass-att | 2 | | 0.64 | 2.17 | 20.3 | 10.1 | 39 |
| Fast-ACV-base-1 | ACV | hourglass | 2 | | 0.71 | 4.61 | 30.7 | 12.1 | 32 |
| **Fast-ACV-E&A** | **Corr8** | **hourglass-E&A** | **1** | **✓** | **0.60** | **2.05** | **12.7** | **8.1** | **30** |
| GwcNet-base | Gwc8 | hourglass | 1 | | 0.67 | 2.20 | 26.2 | 13.3 | **23** |
| **GwcNet-E&A** | **Gwc8** | **hourglass-E&A** | **1** | **✓** | **0.57** | **1.95** | **17.8** | **9.2** | 29 |
| CoEx-base | Corr8 | hourglass-GCE | 1 | | 0.69 | 2.01 | 14.5 | 9.0 | **23** |
| **CoEx-E&A** | **Corr8** | **hourglass-E&A** | **1** | **✓** | **0.60** | **1.93** | **12.5** | **7.1** | 28 |
| CGI-Stereo-base | AFV | hourglass-CGF | 1 | | 0.64 | 1.94 | 13.5 | 6.3 | **28** |
| **CGI-Stereo-E&A** | **AFV** | **hourglass-E&A** | **1** | **✓** | **0.58** | **1.90** | **10.5** | **5.9** | 30 |

We validate the flexibility of the proposed ECF and AGS in four recent SOTA real-time models. Existing context guidance in the above models are replaced by ECF&AGS.

All models were pretrained on Scene Flow and fine-tuned on KITTI. Zero-shot generalization performance was evaluated using the pretrained models. 

The context information fused into the aggregation by ECF makes major contributions to the accuracy improvement, and zero-shot generalization is strengthened by the enhanced geometric information via AGS.



## 🚀 Speed
Timing tests are using the code in **speed.py**, CUDA synchronization included.


## 📈 Table 2. Computational time analysis of each module in Context-Stereo-I and Context-Stereo.

| Module                     | Context-Stereo Time (ms) | Context-Stereo-I Time (ms) |
| :------------------------- | :----------------------: | :-----------------------: |
| Feature Extraction         |            10            |            10             |
| Cost Volume Construction   |            7             |             7             |
| Cost Aggregation  
| ├─ None          |            4             |             4             |
| ├─ ECF     |            7             |             7             |
| ├─ GSC     |            7             |             7             |
| └─ ECF+GSC |            10            |            10             |
| Disparity Iteration        |            -             |            10             |
| Disparity Regression       |            3             |             4             |
| **Full Module**            |          **30**          |          **41**           |
## 🌍 Comparisons with real time methods on real-world data 

![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/Context-Stereo/img/hp.jpg)

The generalization performance of Context-Stereo is evaluated using home-made real-world test data. The image pairs of the real-world roads are captured in Guangzhou City by a binocular camera with a focal length of 6 mm and a baseline distance of 600 mm. The resolution of the image pairs captured is 340×1100. Several regular roads in Guangzhou City are selected as test scenes. In the area where the roads locate, several autonomous driving companies test and run their Robotaxi service.  

The binocular camera is calibrated using OpenCV library and its output image pairs are corrected by the distortion parameters obtained from the calibration and the camera’s parameters. 

Four typical SOTA real-time methods, CoEx, Fast-ACVNet, CGI-Stereo, IINet and RT-IGEV++ are compared with Our methods in the above qualitative results.

## 🥇 Table 3. Comparison of real-time methods on KITTI benchmarks.

| Model | KITTI 2012 3px-noc (%) | KITTI 2012 3px-all (%) | KITTI 2015 D1-bg (%) | KITTI 2015 D1-fg (%) | KITTI 2015 D1-all (%) | Platform | Runtime (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| StereoNet [7] | 4.91 | 6.02 | 4.30 | 7.45 | 4.83 | Titan X | **15** |
| DeepPruner [15] | - | - | 2.32 | 3.91 | 2.59 | RTX 3080 | 50* |
| RTSNet [9] | 2.43 | 2.90 | 2.86 | 6.19 | 3.41 | Tesla P100 | 20 |
| BGNet [13] | 1.77 | 2.15 | 2.07 | 4.74 | 2.51 | RTX 3080 | 26* |
| Fast-ACVNet [8] | 1.68 | 2.13 | 1.82 | 3.93 | 2.17 | RTX 3090 | 39 |
| Fast-ACVNet+ [8] | 1.45 | 1.85 | 1.70 | 3.53 | 2.01 | RTX 3090 | 45 |
| CoEx [16] | 1.55 | 1.93 | 1.79 | 3.82 | 2.13 | RTX 3080 | 23* |
| Ghost-Stereo [14] | 1.45 | 1.80 | 1.71 | 3.77 | 2.05 | RTX 3090 | 37 |
| IINet [25] | 1.81 | 2.21 | 2.02 | 3.39 | 2.25 | RTX 3090 | 26 |
| HITNet [30] | 1.41 | 1.89 | 1.74 | 3.20 | 1.98 | Titan V | 20 |
| CGI-Stereo [18] | 1.41 | 1.76 | 1.66 | 3.38 | 1.94 | RTX 3080 | 28* |
| Light-Stereo-L [20] | 1.55 | 1.87 | 1.78 | **2.64** | 1.93 | RTX 3080 | 45* |
| RT-IGEV++ [42] | 1.29 | 1.68 | 1.48 | 3.37 | 1.79 | RTX 3080 | 48* |
| **Context-Stereo (Ours)** | 1.39 | 1.75 | 1.66 | 3.07 | 1.89 | RTX 3080 | 30 |
| **Context-Stereo-I (Ours)** | **1.26** | **1.66** | **1.47** | 3.05 | **1.73** | RTX 3080 | 41 |

*Speed Were Tested Using Open-Source Code Released by the Authors of Refs. \textbf{Bold}: Best.

## 🛠️ Environment construction

* Create a basic environment and activate it:
```Shell
conda create -n context python=3.8
conda activate context
```
* Ensure the following dependencies are installed:

```bash
pip install torch==2.0.0+cu118 torchvision==0.8.2+cu110 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm==4.66.5
pip install scipy==1.10.1
pip install opencv-python==4.10.0.84
pip install scikit-image==0.21.0
pip install tensorboard==2.13.0
pip install matplotlib==3.7.5
pip install timm==0.9.12
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0
pip install pandas==1.4.4
pip install scikit-learn==1.3.2
pip install einops==0.8.0
pip install h5py==3.11.0
pip install transformers==4.44.2
pip install plotly==5.24.1
pip install open3d==0.19.0
```


## 📂 Dataset 
* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)

Organize the dataset in the following structure:

```
/path/to/dataset/
├── sceneflow/
│   ├── driving__frames_cleanpass/   
│   ├── driving_disparity/   
│   ├── frames_cleanpass/         
│   ├── frames_disparity/   
│   ├── monkaa__frames_cleanpass/   
│   ├── monkaa_disparity/   
├── kitti/
│   ├── training/
│   │   ├── colored_0/
│   │   ├── colored_1/
│   │   ├── disp_occ/
│   │   ├── disp_occ_0/
│   │   ├── image_2/
│   │   ├── image_3/
```


The KITTI dataset used in this project is a mix of **KITTI 2012** and **KITTI 2015** datasets. Ensure that both datasets are properly merged into the `kitti/training` directory.









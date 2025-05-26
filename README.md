# Revisiting Fusion of Context Information into Matching Costs
![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/f111.png)


## 🌟 Environment

The hardware and software environments for training and testing are **NVIDIA RTX 3080 GPU**, Intel i9-12900k (12th gen), 32 GB memory, Ubuntu 22.04 LTS, Python 3.8, PyTorch 2.0.0, CUDA 11.8.

All experiments in the paper including comparisons, ablation experiments in Table 1 and runtime tests are using the above environment.

Pretraining on Scene Flow costs 11.2 h, finetuning on KITTI costs 2.6 h.

## 🏆 KITTI 2012 benchmark
**[Context-Stereo](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=all&result=5e3ff6f4936e065626cf8ebb657bd89f9d1c98d0)** Rank #113.

## 🏆 KITTI 2015 benchmark

**[Context-Stereo](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=19fafc7a0b041ccf935def0c20161f5446976e5f)** Rank #155.

**[Context-Stereo-I](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=fab7f4d69d910af26490342a1aa093a33d4a014c)** Rank #105, the **i**terative version of Context-Stereo. Details of the iterative model will be released shortly in our following publication.

## 📊 Table 1. Performance of ECF and AGS in existing methods on Scene Flow, KITTI 2015 benchmark, Middlebury and ETH3D.

|Method           |Scene Flow EPE (px) |KITTI 2015 D1-all (%)          | Middebury (Zero-shot)     |ETH3D (Zero-shot)  |  Time (ms)              |
|----------------|----------|-----------|----------|---------------|--------------|
|RT-IGEV++      |0.50| 1.79|9.0  |5.7            |72
|RT-IGEV++ (ECF&AGS) |0.44 |1.72  |8.9  |3.9   |76 
|CoEx (GCE)    |0.69|2.01 |14.5  |9.0   |22 
|CoEx (ECF&AGS)|0.60|1.93 |12.5  |7.1   |28 
|CGI-Stereo (CGF)   |0.64|1.94 |13.5  |6.3   |28 
|CGI-Stereo (ECF&AGS)|0.58|1.90  |10.5 |5.9   |30 
|Fast-ACV          |0.64|2.17  |20.3  |10.1  |39 
|Fast-ACV (ECF&AGS)|0.60|2.05  |12.7|8.1  |30 

We validate the flexibility of the proposed ECF and AGS in four recent SOTA real-time models, the fast version of IGEV++, CoEx, CGI-Stereo, and Fast-ACVNet. Existing context guidance in the above models are replaced by ECF&AGS.

All models were pretrained on Scene Flow and fine-tuned on KITTI. Zero-shot generalization performance was evaluated using the pretrained models. 

The context information fused into the aggregation by ECF makes major contributions to the accuracy improvement, and zero-shot generalization is strengthened by the enhanced geometric information via AGS.



## 🚀 Speed
Timing tests are using the code in **speed.py**, CUDA synchronization included.


## 📈 Table 2. Computational time analysis of each module in Context-Stereo.

|Module|  Time (ms)         |
|----------------|----------|
|Feature Extraction     |10|  
|Cost Volume Construction|7|       
|Cost Aggregation (None)|4| 
|Cost Aggregation (ECF)|7| 
|Cost Aggregation (AGS)|7| 
|Cost Aggregation (ECF&AGS)|10| 
|Disparity Regression|3| 
|Full Module|30| 


## 🌍 Comparisons with real time methods on real-world data 

![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/hp2.png)

The generalization performance of Context-Stereo is evaluated using home-made real-world test data. The image pairs of the real-world roads are captured in G****** City by a binocular camera with a focal length of 6 mm and a baseline distance of 600 mm. The resolution of the image pairs captured is 340×1100. Several regular roads in G****** City are selected as test scenes. In the area where the roads locate, several autonomous driving companies test and run their Robotaxi service.  

The binocular camera is calibrated using OpenCV library and its output image pairs are corrected by the distortion parameters obtained from the calibration and the camera’s parameters. 

Four typical SOTA real-time methods, CoEx, Fast-ACVNet, CGI-Stereo, IINet are compared with Context-Stereo in the above qualitative results.

## 🥇 Table 3. Comparison of real-time methods on KITTI benchmarks.

![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/t3.png)

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




## 🎯 Weights 


* [Scene Flow](https://huggingface.co/shidifen12/Context-Stereo/tree/main/)
* [KITTI](https://huggingface.co/shidifen12/Context-Stereo/tree/main/)





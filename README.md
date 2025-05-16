# Revisiting Fusion of Context Information into Matching Costs
![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/f111.png)


## Environment

The hardware and software environments for training and testing are **NVIDIA RTX 3080 GPU**, Intel i9-12900k (12th gen), 32 GB memory, Ubuntu 22.04 LTS, Python 3.8, PyTorch 2.0.0, CUDA 11.8.

All experiments in the paper including comparisons, ablation experiments in Table 1 and runtime tests are using the above enviroment.

Pretraining on Scene Flow costs 11.2 h, finetuning on KITTI costs 2.6 h.


## KITTI 2015 benchmark

**[link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=19fafc7a0b041ccf935def0c20161f5446976e5f)** Rank #156.

## Table 1. Performance of ECF and AGS in existing methods on Scene Flow, KITTI 2015 benchmark, Middlebury and ETH3D.



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



## Speed
Timing tests are using the code in **speed.py**, CUDA synchronization included.


## Table 2. Computational time analysis of each module in Context-Stereo.


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


## Comparisons with real time methods on real-world data 

![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/hp2.png)

The generalization performance of Context-Stereo is evaluated using home-made real-world test data. The image pairs of the real-world roads are captured in G****** City by a binocular camera with a focal length of 6 mm and a baseline distance of 600 mm. The resolution of the image pairs captured is 340×1100. Several regular roads in G****** City are selected as test scenes. In the area where the roads locate, several autonomous driving companies test and run their Robotaxi service.  

The binocular camera is calibrated using OpenCV library and its output image pairs are corrected by the distortion parameters obtained from the calibration and the camera’s parameters. 

Four typical SOTA real-time methods, CoEx, Fast-ACV, CGI-Stere, IINet are compared with Context-Stereo in the above qualititive results.

 

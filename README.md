# Revisiting Fusion of Context Information into Matching Costs
![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/f111.png)


## Environment

The environments for training and testing are **NVIDIA RTX 3080 GPU**, Intel i9-12900k (12th gen), 32 GB memory, Ubuntu 22.04 LTS, Python 3.8, PyTorch 2.0.0, CUDA 11.8.




## Comparisons with real time methods on real-world data 

![imgs](https://github.com/shidifen12/Context-Stereo/blob/main/img/hp2.png)

The generalization performance of proposed network and 4 typical SOTA networks of Fast-ACVNet, CGI-Stereo, IINet  and Coex  are tested using stereo image  pairs captured on real-world roads. The image pairs of the real-world roads are captured in Guangzhou City by a binocular camera with focal length of 6 mm and baseline distance of 600 mm. The resolution of the image pairs captured is 340×1100. An ordinary road in Huangpu District is selected as test scenes. In the area where the road locates, two autonomous driving companies of Apollo and WeRide test and run their Robotaxi service.  The binocular camera is calibrated using OpenCV library and its output image pairs are corrected by the distortion parameters obtained from the calibration and the camera’s parameters. The overall disparity map of the proposed network has accurate distance and proximity relationships, where the predicted vehicles, sidewalks, and trees are with fine contours and continuous details. Compared with the SOTA networks, the object contours obtained by the prediction of the proposed network are more finely and distinguishable, especially for the sky and dense clumps of trees. The accurate disparity prediction of real-world roads by the proposed network facilitates robust 3D perception for autonomous driving.

## KITTI 2015 benchmark

**[link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=19fafc7a0b041ccf935def0c20161f5446976e5f)**

## Performance of ECF and AGS in existing networks on Scene Flow, KITTI 2015 benchmark, Middlebury and ETH3D.



|Method           |Scene Flow EPE (px) |KITTI 2015 D1-all (%)          | Middebury (Zero-shot)     |ETH3D (Zero-shot)  |  Time (ms)              |
|----------------|----------|-----------|----------|---------------|--------------|
|RT-IGEV++      |0.50| 1.79|9.02  |5.66            |72
|RT-IGEV++ (ECF&AGS) |0.44 |1.72  |8.92  |3.91   |76 
|CoEx (GCE)    |0.69|2.01 |14.5  |9.0   |22 
|CoEx (ECF&AGS)|0.60|1.93 |12.5  |7.1   |28 
|CGI-Stereo (CGF)   |0.64|1.94 |13.5  |6.3   |28 
|CGI-Stereo (ECF&AGS)|0.58|1.90  |10.5 |5.9   |30 
|Fast-ACV   |0.64|2.17 |20.3  |10.1  |39 
|Fast-ACV (ECF&AGS)|0.60|2.05  | |   |30 

We validate the proposed ECF and AGS to fuse context information and builds learnable connection in the fast verson of IGEV++ model，Coex，  CGI-Stereo， and Fast-ACVNet,. Existing module of context fusion in there networks  is replaced by ECF and AGS. As the results , pretrained (Scene Flow) and fine-tuned (KITTI) results are improved.  Moreover, zero-shot generalization is significantly improved by ECF and AGS. 


## Computational time analysis of each module in our model.


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

# Task-related Saliency Network For Few-shot learning
This is an official implementation in Tensorflow of TRSN.


![](image_screenshot_10.11.2021.png)
![](q_show_image_screenshot_10.11.2021.png)

## Abstract
An essential cue of human wisdom in the few-shot classification task is that they can find the task-related targets by a glimpse of support images. Thus, we propose to divide the tackling of few-shot classification into three phases including Modeling, Analysing and Matching. In the modeling phase, we introduce a Saliency Sensitive Module (SSM), which is an inexact supervision task jointly trained with a standard multi-class classification task. SSM not only promote the representation ability of feature embedding, but also can locate the task-related saliency features. Therefore, we propose a self-training based Task-related Saliency Network (TRSN) to the learning of locating the saliency objects produced by SSM. In the analysing phase, we utilize TRSN to find out the task-related features. In the matching phase, we make the representation fused with task-related features to help samples matching their most related proto. We conduct extensive experiments on 5-way 1-shot and 5-way 5-shot settings to evaluate the proposed method. Results show that our method achieves a consistent performance gain on benchmarks. Moreover, our method is state-of-the-art on the fine-grained few-shot classification of CUB.
## Performance
![](miniimagenet.png)
![](CUB.png)

### Environment
- CUDA == 10.1
- Tensorflow == 2.2.0 

# FER
## Dataset
We are using LFW emotion dataset. The dataset download link: [LFW-emotion-dataset](https://drive.google.com/file/d/1vM3qHpZ6PcrU9UGwEbnAYl-gmh-rOw2r/view?usp=sharing)


The related paper can be found from here: [Facial Expression Recognition with the advent of face masks](https://www.researchgate.net/publication/346519054_Facial_Expression_Recognition_with_the_advent_of_face_masks)

LFW emotion dataset is annotated based on [LFW](http://vis-www.cs.umass.edu/lfw/) (Labeled Faces in the Wild).

This dataset consists of **three** parts and can be used for the following study:

* **LFW-FER:**  LFW dataset annoteted manually for facial expression recoginition study.
* **M-LFW_FER:**  LFW dataset processed by automatic wearing face mask method for masked facial expression recoginition study.
* **M-LFW_FER-face-cut:**  M-LFW_FER dataset processed by automatic cut the face only.


<!-- timm == 0.6.7
AttributeError: 'README.md
LFW emotion datasetEfficientNet' object has no attribute 'act1' in forward computation

enet_b0_7, timm = 0.4.5 (OK). But new version
AttributeError: 'EfficientNet' object has no attribute 'grad_checkpointing' -->

## State-of-the-art method 
We would summarize all the state-of-the-art and trendy methods for FER (Facial expression recoginition) task with the highest accuracy for different dataset such as AffectNet, RAF-DB, etc according to the benchmark which draws on [paperswithcode](https://paperswithcode.com/task/facial-expression-recognition).

|Model|Paper name|Paper|Github|Status|
|--|--|--|--|--|--|
|Multi-task EfficientNet-B2|Classifying emotions and engagement in online learning based on a single facial expression recognition neural network (IEEE 2022)|[Link](https://ieeexplore.ieee.org/document/9815154)|[Link](https://github.com/HSE-asavchenko/face-emotion-recognition)|:white_check_mark:|
|DAN|Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition (2021)|[Link](https://arxiv.org/pdf/2109.07270v4.pdf)|[Link](https://github.com/yaoing/dan)|In process|
<!-- |FAN|Frame attention networks for facial expression recognition in videos (2019)|[Link](https://arxiv.org/pdf/1907.00193v2.pdf)|[Link](https://github.com/Open-Debin/Emotion-FAN)|In process|
|DeepEmotion|Deep-Emotion: Facial Expression Recognition Using Attentional Convolutional Network (2019)|[Link](https://arxiv.org/pdf/1902.01019v1.pdf)|[Link](https://github.com/omarsayed7/Deep-Emotion)|No pre-trained model|
|Ensemble with Shared Representations
(ESR-9)|Efficient Facial Feature Learning with Wide Ensemble-based Convolutional Neural Networks (2020)|[Link](https://arxiv.org/pdf/2001.06338v1.pdf)|[Link](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks)|10 pre-trained models|
|RAN|Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition (2019)|[Link](https://arxiv.org/pdf/1905.04075v2.pdf)|[Link](https://github.com/kaiwang960112/Challenge-condition-FER-dataset)|1 pre-trained model|
|Ours (VGG-F)|Pre-training strategies and datasets for facial representation learning (2021)|[Link](https://arxiv.org/pdf/2103.16554v2.pdf)|[Link](https://github.com/1adrianb/unsupervised-face-representation)|Pending|
|...|Challenges in Representation Learning: A report on three machine learning contests (2013)|[Link](https://arxiv.org/pdf/1307.0414v1.pdf)|[Link](https://github.com/phamquiluan/ResidualMaskingNetwork)|Pending| -->

<!-- In process
:white_check_mark: -->


## Train the pre-trained model with dataset
We use the pre-trained model of state-of-the-art model for FER task for different dataset, change the last layer classifier into 3 nodes instead of 7 or 8 original expressions in order to predict only 3 expression neutral, possitive and negative.

|Pre-trained Model|Model|LFW-FER|M-LFW-FER|M-LFW-FER-face-cut|
|--|--|--|--|--|
|enet_b0_7.pt|Multi-task EfficientNet-B2|87.4676|38.6147|71.4757|
|enet_b0_8_best_afew.pt|Multi-task EfficientNet-B2|87.3812|41.5584|76.5870|
|enet_b0_8_best_vgaf.pt|Multi-task EfficientNet-B2|86.6033|57.2294|75.5153|
|enet_b0_8_va_mtl.pt|Multi-task EfficientNet-B2|87.6404|47.0996|76.8343|
|enet_b2_8.pt|Multi-task EfficientNet-B2|87.4676|56.1039|76.9167

<!-- 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classifying-emotions-and-engagement-in-online/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=classifying-emotions-and-engagement-in-online)
     -->

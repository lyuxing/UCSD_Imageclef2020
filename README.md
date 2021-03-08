# UCSD_Imageclef2020


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


# Abstract
We proposed an AI model with laterality-reduction 3D CBAM Resnet and balanced-sampler strategy to detect and characterize of tuberculosis and the evaluation of lesion characteristics as a solution for a classification of tuberculosis findings. Detection and characterization of tuberculosis and the evaluation of lesion characteristics are challenging. In an effort to provide a solution for a classification task of tuberculosis findings, we proposed a laterality-reduction 3D AI model with attention mechanism and balanced-sampler strategy. With proper usage of both provided masks, each side of the lung was cropped, masked, and rearranged so that laterality could be neglected, and dataset size doubled. Balanced sampler in each batch sampler was also used in this study to address the data imbalance problem. CBAM was used to add an attention mechanism in each block of the Res-net to further improve the performance of the CNN.


# Environment

- python 3.6.2
- numpy 1.14.3
- pytorch 0.4.0
- pillow 3.4.2
- opencv 3.1.0
- skimage 0.16.2
- pandas 1.0.3
- nibabel 3.0.2

# Run

## Example Data

TB2020 public dataset link:
https://www.imageclef.org/2020/medical/tuberculosis

Some testing dataset examples: 
https://drive.google.com/drive/folders/1fPd1PRRa_cxBCyEkH2XOMpkpvSKoQo6S?usp=sharing

Please download the data to your local path: [data_path]

## Image preparation and Network architecture

![image](https://github.com/lyuxing/UCSD_Imageclef2020/blob/main/img/Figure%202%20network%20architecture.png)

![image](https://github.com/lyuxing/UCSD_Imageclef2020/blob/main/img/Figure%203%20image%20preprocessing.png)

## Pretrained model path

You could download the model to your local path: [model_path]

https://drive.google.com/file/d/1wduk9us3OH1WWJ6mgAwPqjekdUd8OGAg/view?usp=sharing

## Model Inference

To test the performance on test dataset, run:
`python3 inference_args.py --img_pth=<image dir> 
                           --msk1_pth=<mask1 dir> 
                           --msk2_pth=<mask2 dir> 
                           --img_id = <image id>
                           --model_path = <model path>'
The results of the model will be printed. 

# Contact

Xing Lu: lvxingvir@gmail.com

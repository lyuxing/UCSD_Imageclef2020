# UCSD_Imageclef2020
For ImageCLEFmed Tuberculosis competition, the 2ed place. 
Paper link: http://ceur-ws.org/Vol-2696/paper_70.pdf

# Introduction
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

## Dataset

TB2020 public dataset link:
https://www.imageclef.org/2020/medical/tuberculosis

Some testing dataset examples: 
https://drive.google.com/drive/folders/1fPd1PRRa_cxBCyEkH2XOMpkpvSKoQo6S?usp=sharing

Please download the data to your local path: [data_path]

## Network architecture and Image preparation

![image](https://github.com/lyuxing/UCSD_Imageclef2020/blob/main/img/Figure%202%20network%20architecture.png)

A 3D convolutional block attention module (CBAM)-Resnet was designed to train the model for 3-class binary classification based on the PyTorch framework. A standard 3D-resnet34 was used as the convolutional neural network backbone, with three fc layers to be the classifier. CBAM was used to implement channel and spatial at-tention mechanisms for each block of the Resnet. Sigmoid was used as the activation function for binary classification.

![image](https://github.com/lyuxing/UCSD_Imageclef2020/blob/main/img/Figure%203%20image%20preprocessing.png)



## Pretrained model path

You could download the model to your local path: [model_path]

https://drive.google.com/file/d/1wduk9us3OH1WWJ6mgAwPqjekdUd8OGAg/view?usp=sharing

## Model Inference

To test the performance on test dataset, run:

`python3 inference_args.py --img_pth='<image dir> '
                           --msk1_pth='<mask1 dir> '
                           --msk2_pth='<mask2 dir> '
                           --img_id = '<image id>'
                           --model_path = '<model path>''
                           
The results of the model will be printed. 

# Contact

Paper link: http://ceur-ws.org/Vol-2696/paper_70.pdf

Xing Lu: lvxingvir@gmail.com
Gentili Amilcare: agentili@health.ucsd.edu
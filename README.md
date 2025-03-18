Dislocation Detection in Grain Boundaries

This repository contains an implementation of dislocation probability estimation and localization within grain boundaries using two different deep learning architectures: UNet and Vision Transformer (ViT-B/16). The models are designed to predict the probability of dislocation presence and provide their (x, y) positions in given images.

UNet (U-Net segmentation model)
The UNet model is based on an encoder-decoder architecture with skip connections. Originally designed for segmentation, it is here adapted to dislocation localization.

UNet key points:
Encoder with 5 levels of convolutions.
Decoder with 5 levels upsampling layers.
Prediction of dislocation probabilities and coordinates.

ViT-B/16 (Vision Transformer - ViT)
The ViT-B/16 model is a pre-trained ImageNet Transformer modified to perform a dislocation localization task.

ViT-B/16 key points:
ImageNet pre-trained model.
Replacement of the classification head to generate (x,y) coordinates and probabilities (p1,p0).
Ability to capture long-range relationships in the image.
12 encoder layer. Architecture of one encoder layer : - Multi-Head Self-Attention (MSA) layer
                                                      - MLP (Multi-Layer Perceptron)
                                                      - Layer Normalization and Skip Connections
86 million parameters in ViT-B/16 version.


Results :
          - The ViT model offers the best accuracy but is more computationally expensive.
          - The UNet model is fast but less accurate than the ViT model.

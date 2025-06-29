# CNN Variants

This document outlines various popular Convolutional Neural Network (CNN) architectures and their key characteristics.

## LeNet-5
- **Year:** 1998
- **Key Features:** One of the earliest successful CNNs, designed for handwritten digit recognition. It introduced concepts like convolutional layers, pooling layers, and fully connected layers.

## AlexNet
- **Year:** 2012
- **Key Features:** Broke records in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Utilized ReLU activation functions, dropout for regularization, and data augmentation. Showcased the power of deep CNNs.

## VGGNet
- **Year:** 2014
- **Key Features:** Emphasized the use of very small (3x3) convolutional filters stacked in multiple layers to increase depth. Demonstrated that depth is a critical component for good performance.

## GoogLeNet (Inception)
- **Year:** 2014
- **Key Features:** Introduced the "Inception module" to allow the network to learn multiple feature extraction paths in parallel (1x1, 3x3, 5x5 convolutions, and pooling). This reduced the number of parameters while increasing network depth and width.

## ResNet (Residual Networks)
- **Year:** 2015
- **Key Features:** Introduced "residual connections" (skip connections) to allow gradients to flow directly through the network, addressing the vanishing gradient problem in very deep networks. Enabled the training of extremely deep CNNs (e.g., 152 layers).

## DenseNet (Densely Connected Convolutional Networks)
- **Year:** 2017
- **Key Features:** Each layer is directly connected to every other layer in a feed-forward fashion. This promotes feature reuse and reduces the number of parameters, leading to more compact models.

## MobileNet
- **Year:** 2017
- **Key Features:** Designed for mobile and embedded vision applications. Utilizes depthwise separable convolutions to significantly reduce computational cost and model size while maintaining accuracy.

## EfficientNet
- **Year:** 2019
- **Key Features:** Systematically scales network depth, width, and resolution using a compound scaling method. Achieves state-of-the-art accuracy with significantly fewer parameters and FLOPs compared to other models.
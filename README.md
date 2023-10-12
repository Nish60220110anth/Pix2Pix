# Pix2Pix

## Introduction

Pix2Pix is a supervised GAN algorithm that is used to generate images from one domain to another. For example, it can be used to convert images from day to night, or from sketches to real images. This implementation is based on the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004).

## Idea

We train the model on paired images from two domains. The model learns to translate images from one domain to another. The model consists of two parts: the generator and the discriminator. Generator learns to generate images from the source domain, while the discriminator learns to distinguish between real and fake images. Generator is trained to fool the discriminator, while the discriminator is trained to distinguish between real and fake images.

Discriminator is sent with real images from the source domain and fake images from the generator.

## Loss

The loss function consists of two parts: adversarial loss and L1 loss. Adversarial loss is the standard GAN loss. L1 loss is the mean absolute error between the generated image and the target image. The total loss is the sum of the two losses.

## Issues and Solutions

### 1. Use low learning rate for the discriminator

## Usage

Check `check.py` for an example of how to use the model.
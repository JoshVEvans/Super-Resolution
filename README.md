# Single-Image Super Resolution (SISR)
## What is Super Resolution?
SISR is the process of upscaling a low resolution image to high resolution.

## Examples
![alt text](evaluation/Combined/000000000029.jpg)
The images from left to right are original, interpolated through bicubic upscaling, and prediction of the neural network. To see a higher version, click on the image.
![alt text](evaluation/Combined/000000001300.jpg)

## Network Architecture:
The model consists of 24 residual blocks. Each block contains a Convolution, Relu, Convolution, Add layers.

            x
		|\
		| \
		|  conv2d
		|  activation
		|  conv2d
            |  (multiply scaling)
		| /
		|/
		+ (residual scaling)
		|
		result
![alt text](data/model_small.png)
An image of the complete model is towards the bottom of this page.

## How do you use this model?
Put images you want to upscale in side the '**inference/original**' directory. Run output.py and the results will be written into the '**inference/output**' directory. It should take a couple of seconds to run the model for each image inside the input directory.

## Complete Model Architecture.
![alt text](data/model_large.png)


### Author
##### Joshua Evans - [github/JoshVEvans](https://github.com/JoshVEvans)

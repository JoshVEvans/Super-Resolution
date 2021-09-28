# Single-Image Super Resolution (SISR)
## What is Super Resolution?
Super Resolution is the process of upscaling an image from low to high resolution. Many methods are commonly used for SR, such as nearest-neighbor interpolation, bicubic interpolation, and Lanczos upsampling [(here)](https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms). Deep learning methods improve previous techniques by applying common patterns found within high-resolution training data onto the low-resolution image during inference.

## Examples
![alt text](evaluation/Combined/000000000029.jpg)
##### *From left to right: original, interpolated(Nearest Neighbor upscaling), and prediction(from neural network)*<br />
To see a higher quality version, **[click](https://github.com/JoshVEvans/Super-Resolution/tree/master/evaluation/Combined) on the images**. In some cases, the upscaled image looks even better than the original!
![alt text](evaluation/Combined/000000001300.jpg)

## Reasearch and Development
How did I get to this model architecture? I initially started with a very early architecture known as Single-Image Convolutional Neural Network ([SRCNN](https://arxiv.org/pdf/1501.00092.pdf)). This architecture consists of 2 hidden layers and a reconstruction layer as an output.
![alt text](md_images/srcnn.png)
The next model I tried implementing was Very Deep Super-Resolution ([VDSR](https://arxiv.org/pdf/1511.04587.pdf)). This model improves upon the original SRCNN by adding a global skip connection, thus making upscaling much easier. Essentially, the network doesn't need to reconstruct the image entirely and instead needs to reconstruct the difference (the residual) between the high and low-resolution image.
<br />
My implementation uses the idea of skip connections found within VDSR and implements both global and local connections using an `Add` Layer. Since this model is quite deep, I also implemented a smaller model that uses `Concatenate` layers instead of using `Add` Layers.

## Network Architecture:
#### The core of this model is in the residual blocks.

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

#### Uses `Add` layers to add residuals
![alt text](md_images/model_large_residuals.png)

#### This is the smaller model that uses `Concatenate` layers to add residuals
![alt text](md_images/model_small.png)

An image of the complete model is towards the bottom of this page.

## How do you use this model?
Put low-resolution images to upscale inside the '**inference/original**' directory. Run output.py, and the results will be written into the '**inference/output**' directory. It should take a couple of seconds to run the model for each image inside the input directory.

## How can you train your own model?
The model is instantiated within [`network.py`](https://github.com/JoshVEvans/Super-Resolution/blob/master/network.py). You can play around with hyper-parameters there. First, to train the model, delete the images currently within `data/` put your training image data within that file - I recommend the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Finally, mess with hyper-parameters in [`train.py`](https://github.com/JoshVEvans/Super-Resolution/blob/master/train.py) and run `train.py`. If you’re training on weaker hardware, I’d recommend lowering the `batch_size` below the currently set ***8*** images. Also, decrease the number of (`residual blocks`) from `24 to 9` and reduce the number of filters (`num_filters`) from `128 to 64`.

## More Examples:
#### Set 5 Evaluation Set:
Images Left to Right: Original, Nearest Neighbor, Predicted.
![alt text](evaluation/Combined/baboon.png)
![alt text](evaluation/Combined/baby.png)
![alt text](evaluation/Combined/butterfly.png)
![alt text](evaluation/Combined/comic.png)

## Complete Model Architecture:

![alt text](md_images/model_large.png)

### Hardware - Training Statistics
##### Trained on 3070 ti
###### Batch Size: 8
###### Training Image Size: 64x64
##### Training Time ~22hrs

### Author
##### Joshua Evans - [github/JoshVEvans](https://github.com/JoshVEvans)

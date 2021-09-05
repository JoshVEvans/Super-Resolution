# Single-Image Super Resolution (SISR)
## What is Super Resolution?
Super Resolution is the process of upscaling a low-resolution image into a high resolution. Many methods are commonly used for SR, such as nearest-neighbor interpolation, bicubic interpolation, and Lanczos upsampling [(here)](https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms). Deep learning methods improve previous techniques by applying common patterns found within high-resolution training data in the low-resolution image during inference.

## Examples
![alt text](evaluation/Combined/000000000029.jpg)
##### *From left to right: original, interpolated(Nearest Neighbor upscaling), and prediction(from neural network)*<br />
To see a higher quality version, **[click](https://github.com/JoshVEvans/Super-Resolution/tree/master/evaluation/Combined) on the images**. In some cases, the upscaled image looks even better than the original!
![alt text](evaluation/Combined/000000001300.jpg)

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
		result'

![alt text](data/model_small.png)

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

## Complete Model Architecture.

![alt text](data/model_large.png)

### Hardware - Training Statistics
##### Trained on 3070 ti
###### Batch Size: 8
###### Training Image Size: 64x64
##### Training Time ~22hrs

### Author
##### Joshua Evans - [github/JoshVEvans](https://github.com/JoshVEvans)

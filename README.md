This code will allow you to collect the activations of the PixelNerf renderer. There are 5 layers (0-4). Run this:

```
from eval.activations_pixelnerf import PixelNerf

model = PixelNerf()
layer_num = [Number from 0-4]
inputs = [a list of the image names]
IMG_PTH = [a list of the path of the images or a string if its the same for all]
model.compute_activation(layer_num, inputs, IMG_PTH)
```

To use the activations, run `model.all_activations`. Furthermore, the activations are shaped in 4D array, but if flattened is preferred, that code should be added.  

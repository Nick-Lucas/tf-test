# tf-test

Some test code you can check-out to test your GPU configuration

## Dependencies

Code is intended for testing existing environments, so no requirements.txt, but all you need is:

* numpy
* tensorflow
* tensorflow-gpu (if testing GPUs)

## Test if GPU is detected

`python check-gpu.py`

## Test if a simple dense-layered NN can run

`python train-dense.py`

## Test if an mnist conv-layered NN can run

`python train-conv.py`

Code adapted from this [medium article](https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f)


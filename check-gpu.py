import tensorflow as tf

print("Is GPU available: ", tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
))

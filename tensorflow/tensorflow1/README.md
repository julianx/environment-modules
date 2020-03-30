== TF1 test

    (/opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda) [train203@gpu047 tf1_test_urbanic]$ python CNN_original.py
    WARNING:tensorflow:From CNN_original.py:5: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./train-images-idx3-ubyte.gz
    WARNING:tensorflow:From /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Extracting ./t10k-images-idx3-ubyte.gz
    Extracting ./t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages/tensorflow_core/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From CNN_original.py:7: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

    WARNING:tensorflow:From CNN_original.py:12: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.

    WARNING:tensorflow:From CNN_original.py:15: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

    WARNING:tensorflow:From CNN_original.py:30: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    WARNING:tensorflow:From CNN_original.py:34: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

    WARNING:tensorflow:From CNN_original.py:38: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.

    2020-03-28 16:49:12.888992: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    2020-03-28 16:49:12.902633: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095054999 Hz
    2020-03-28 16:49:12.908183: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fc95667c70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-03-28 16:49:12.908211: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-03-28 16:49:12.910611: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
OK  2020-03-28 16:49:12.933784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
    name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
    pciBusID: 0000:81:00.0
    2020-03-28 16:49:12.934391: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2020-03-28 16:49:12.936745: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2020-03-28 16:49:12.939131: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2020-03-28 16:49:12.939965: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2020-03-28 16:49:12.942702: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2020-03-28 16:49:12.944816: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2020-03-28 16:49:12.950356: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-28 16:49:12.952976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2020-03-28 16:49:12.953013: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2020-03-28 16:49:13.066894: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-03-28 16:49:13.066944: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
    2020-03-28 16:49:13.066957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
    2020-03-28 16:49:13.070984: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15216 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:81:00.0, compute capability: 6.0)
    2020-03-28 16:49:13.073434: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fc961fd5e0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-03-28 16:49:13.073457: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla P100-PCIE-16GB, Compute Capability 6.0
    WARNING:tensorflow:From CNN_original.py:40: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

    2020-03-28 16:49:13.878400: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2020-03-28 16:49:14.120759: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    step 0, training accuracy 0.16
    step 100, training accuracy 0.8
    step 200, training accuracy 0.92
    step 300, training accuracy 0.86
    step 400, training accuracy 0.9
    step 500, training accuracy 0.96
    step 600, training accuracy 0.92
    step 700, training accuracy 0.98
    step 800, training accuracy 0.96
    step 900, training accuracy 0.94
    step 1000, training accuracy 0.98
    step 1100, training accuracy 0.98
    step 1200, training accuracy 1
    step 1300, training accuracy 1
    step 1400, training accuracy 1
    step 1500, training accuracy 0.98
    step 1600, training accuracy 0.92
    step 1700, training accuracy 1
    step 1800, training accuracy 0.96
    step 1900, training accuracy 0.98
    step 2000, training accuracy 1
    step 2100, training accuracy 1
    step 2200, training accuracy 0.98
    step 2300, training accuracy 0.98
    step 2400, training accuracy 0.98
    step 2500, training accuracy 1
    step 2600, training accuracy 1
    step 2700, training accuracy 1
    step 2800, training accuracy 0.96
    step 2900, training accuracy 0.96
    step 3000, training accuracy 1
    step 3100, training accuracy 1
    step 3200, training accuracy 0.98
    step 3300, training accuracy 1
    step 3400, training accuracy 0.98
    step 3500, training accuracy 0.98
    step 3600, training accuracy 0.96
    step 3700, training accuracy 1
    step 3800, training accuracy 0.98
    step 3900, training accuracy 0.98
    step 4000, training accuracy 1
    step 4100, training accuracy 1
    step 4200, training accuracy 1
    step 4300, training accuracy 0.96
    step 4400, training accuracy 1
    step 4500, training accuracy 1
    step 4600, training accuracy 0.98
    step 4700, training accuracy 1
    step 4800, training accuracy 1
    step 4900, training accuracy 0.94
    step 5000, training accuracy 0.98
    step 5100, training accuracy 1
    step 5200, training accuracy 0.98
    step 5300, training accuracy 1
    step 5400, training accuracy 1
    step 5500, training accuracy 0.98
    step 5600, training accuracy 1
    step 5700, training accuracy 1
    step 5800, training accuracy 1
    step 5900, training accuracy 1
    step 6000, training accuracy 1
    step 6100, training accuracy 0.98
    step 6200, training accuracy 1
    step 6300, training accuracy 1
    step 6400, training accuracy 1
    step 6500, training accuracy 1
    step 6600, training accuracy 1
    step 6700, training accuracy 1
    step 6800, training accuracy 0.98
    step 6900, training accuracy 1
    step 7000, training accuracy 1
    step 7100, training accuracy 0.98
    step 7200, training accuracy 1
    step 7300, training accuracy 1
    step 7400, training accuracy 1
    step 7500, training accuracy 0.96
    step 7600, training accuracy 0.98
    step 7700, training accuracy 1
    step 7800, training accuracy 1
    step 7900, training accuracy 1
    step 8000, training accuracy 0.98
    step 8100, training accuracy 1
    step 8200, training accuracy 1
    step 8300, training accuracy 1
    step 8400, training accuracy 0.98
    step 8500, training accuracy 1
    step 8600, training accuracy 1
    step 8700, training accuracy 0.98
    step 8800, training accuracy 0.98
    step 8900, training accuracy 1
    step 9000, training accuracy 1
    step 9100, training accuracy 1
    step 9200, training accuracy 1
    step 9300, training accuracy 1
    step 9400, training accuracy 1
    step 9500, training accuracy 0.98
    step 9600, training accuracy 0.98
    step 9700, training accuracy 1
    step 9800, training accuracy 1
    step 9900, training accuracy 1
    step 10000, training accuracy 1
    step 10100, training accuracy 1
    step 10200, training accuracy 0.98
    step 10300, training accuracy 1
    step 10400, training accuracy 1
    step 10500, training accuracy 1
    step 10600, training accuracy 1
    step 10700, training accuracy 1
    step 10800, training accuracy 1
    step 10900, training accuracy 1
    step 11000, training accuracy 1
    step 11100, training accuracy 1
    step 11200, training accuracy 1
    step 11300, training accuracy 1
    step 11400, training accuracy 1
    step 11500, training accuracy 1
    step 11600, training accuracy 1
    step 11700, training accuracy 0.96
    step 11800, training accuracy 1
    step 11900, training accuracy 0.98
    step 12000, training accuracy 1
    step 12100, training accuracy 1
    step 12200, training accuracy 1
    step 12300, training accuracy 1
    step 12400, training accuracy 1
    step 12500, training accuracy 1
    step 12600, training accuracy 1
    step 12700, training accuracy 1
    step 12800, training accuracy 1
    step 12900, training accuracy 1
    step 13000, training accuracy 1
    step 13100, training accuracy 1
    step 13200, training accuracy 1
    step 13300, training accuracy 1
    step 13400, training accuracy 1
    step 13500, training accuracy 1
    step 13600, training accuracy 1
    step 13700, training accuracy 1
    step 13800, training accuracy 1
    step 13900, training accuracy 1
    step 14000, training accuracy 1
    step 14100, training accuracy 0.98
    step 14200, training accuracy 1
    step 14300, training accuracy 1
    step 14400, training accuracy 1
    step 14500, training accuracy 1
    step 14600, training accuracy 1
    step 14700, training accuracy 1
    step 14800, training accuracy 1
    step 14900, training accuracy 1
    step 15000, training accuracy 1
    step 15100, training accuracy 1
    step 15200, training accuracy 1
    step 15300, training accuracy 1
    step 15400, training accuracy 1
    step 15500, training accuracy 1
    step 15600, training accuracy 1
    step 15700, training accuracy 1
    step 15800, training accuracy 1
    step 15900, training accuracy 1
    step 16000, training accuracy 1
    step 16100, training accuracy 0.96
    step 16200, training accuracy 1
    step 16300, training accuracy 1
    step 16400, training accuracy 1
    step 16500, training accuracy 1
    step 16600, training accuracy 1
    step 16700, training accuracy 0.96
    step 16800, training accuracy 1
    step 16900, training accuracy 1
    step 17000, training accuracy 1
    step 17100, training accuracy 1
    step 17200, training accuracy 1
    step 17300, training accuracy 1
    step 17400, training accuracy 1
    step 17500, training accuracy 1
    step 17600, training accuracy 1
    step 17700, training accuracy 1
    step 17800, training accuracy 1
    step 17900, training accuracy 1
    step 18000, training accuracy 1
    step 18100, training accuracy 1
    step 18200, training accuracy 1
    step 18300, training accuracy 0.98
    step 18400, training accuracy 0.98
    step 18500, training accuracy 1
    step 18600, training accuracy 1
    step 18700, training accuracy 1
    step 18800, training accuracy 1
    step 18900, training accuracy 1
    step 19000, training accuracy 1
    step 19100, training accuracy 0.98
    step 19200, training accuracy 1
    step 19300, training accuracy 1
    step 19400, training accuracy 1
    step 19500, training accuracy 1
    step 19600, training accuracy 1
    step 19700, training accuracy 1
    step 19800, training accuracy 1
    step 19900, training accuracy 1
OK  test accuracy 0.9927
    (/opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda) [train203@gpu047 tf1_test_urbanic]$ module list
    Currently Loaded Modulefiles:
    1) psc_path/1.1                       3) intel/19.5                         5) gcc/5.3.0                          7) tensorflow/1.15_py3_gpu_anaconda
    2) slurm/default                      4) xdusage/2.1-1                      6) cuda/10.1
    (/opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda) [train203@gpu047 tf1_test_urbanic]$


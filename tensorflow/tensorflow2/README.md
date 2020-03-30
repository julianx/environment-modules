## TensorRT warnings
When installing TensorFlow 2, not having TensorRT will trigger the following warnings when loading the tensorflow module:

2020-03-25 17:40:36.636088: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/packages/mkl-dnn/external/mklml_lnx_2018.0.1.20171227/lib:/opt/packages/python/gnu_openmpi/3.6.4_np1.14.5_pip20.0.2/lib:/opt/packages/phdf5/gnu_openmpi/1.10.2/lib:/usr/mpi/gcc/openmpi-2.1.2-hfi/lib64:/opt/packages/cuda/10.1/lib64:/opt/packages/gcc/5.3.0/lib64:/opt/packages/gcc/5.3.0/lib
2020-03-25 17:40:36.636246: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/packages/mkl-dnn/external/mklml_lnx_2018.0.1.20171227/lib:/opt/packages/python/gnu_openmpi/3.6.4_np1.14.5_pip20.0.2/lib:/opt/packages/phdf5/gnu_openmpi/1.10.2/lib:/usr/mpi/gcc/openmpi-2.1.2-hfi/lib64:/opt/packages/cuda/10.1/lib64:/opt/packages/gcc/5.3.0/lib64:/opt/packages/gcc/5.3.0/lib
2020-03-25 17:40:36.636262: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.

### For fixing the warnings, install TensorRT:

https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt_304/tensorrt-install-guide/index.html#installing-tar
    
    TensorRT install
        lsb_release -a
            Description:	CentOS Linux release 7.6.1810 (Core)
        # File downloaded
            TensorRT-7.0.0.11.CentOS-7.6.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz
                drwxr-xr-x root/root         0 2019-12-16 21:26 TensorRT-7.0.0.11/
                lrwxrwxrwx root/root         0 2019-12-16 21:17 TensorRT-7.0.0.11/bin -> targets/x86_64-linux-gnu/bin
                lrwxrwxrwx root/root         0 2019-12-16 21:17 TensorRT-7.0.0.11/lib -> targets/x86_64-linux-gnu/lib
                drwxr-xr-x root/root         0 2019-12-16 21:26 TensorRT-7.0.0.11/targets/
                drwxr-xr-x root/root         0 2019-12-16 21:17 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/
                drwxr-xr-x root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/bin/
                -rwxr-xr-x root/root    268296 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/bin/trtexec
                -rwxr-xr-x root/root       352 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/bin/giexec
                drwxr-xr-x root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvcaffe_parser.a -> libnvparsers_static.a
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvcaffe_parser.so.7.0.0 -> libnvparsers.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvcaffe_parser.so.7 -> libnvparsers.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvcaffe_parser.so -> libnvparsers.so.7.0.0
                -rw-r--r-- root/root   4434146 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvparsers_static.a
                -rwxr-xr-x root/root   3367320 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvparsers.so.7.0.0
                -rw-r--r-- root/root   1344920 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libprotobuf-lite.a
                -rw-r--r-- root/root  15956258 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libprotobuf.a
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvparsers.so.7 -> libnvparsers.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvparsers.so -> libnvparsers.so.7.0.0
                -rw-r--r-- root/root 262839168 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer_static.a
                -rwxr-xr-x root/root 228711152 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer.so.7 -> libnvinfer.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer.so -> libnvinfer.so.7.0.0
                -rw-r--r-- root/root   7235410 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer_plugin_static.a
                -rwxr-xr-x root/root   5606064 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.7 -> libnvinfer_plugin.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so -> libnvinfer_plugin.so.7.0.0
                -rwxr-xr-x root/root   2494744 2019-12-16 21:17 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvonnxparser.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvonnxparser.so.7 -> libnvonnxparser.so.7.0.0
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvonnxparser.so -> libnvonnxparser.so.7
                -rw-r--r-- root/root   1400072 2019-12-16 21:12 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libnvonnxparser_static.a
                drwxr-xr-x root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/stubs/
                -rwxr-xr-x root/root    513888 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/stubs/libnvinfer.so
                -rwxr-xr-x root/root      5984 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/stubs/libnvparsers.so
                -rwxr-xr-x root/root    137056 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/stubs/libnvinfer_plugin.so
                -rwxr-xr-x root/root      5984 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/stubs/libnvonnxparser.so
                -rw-r--r-- root/root      2702 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/stubs/libnvrtc_static.a
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin.so -> libmyelin.so.1
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin.so.1 -> libmyelin.so.1.0.0
                -rwxr-xr-x root/root   6261056 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin.so.1.0.0
                -rw-r--r-- root/root  11570388 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin_compiler_static.a
                -rw-r--r-- root/root   8424872 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin_executor_static.a
                -rw-r--r-- root/root    339120 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin_pattern_library_static.a
                -rw-r--r-- root/root    182014 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libmyelin_pattern_runtime_static.a
                -rw-r--r-- root/root    457436 2019-12-16 21:21 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib/libonnx_proto.a
                lrwxrwxrwx root/root         0 2019-12-16 21:17 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/include -> ../../include
                lrwxrwxrwx root/root         0 2019-12-16 21:17 TensorRT-7.0.0.11/targets/x86_64-linux-gnu/samples -> ../../samples
                drwxr-xr-x root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/samples/
                [...]
                drwxr-xr-x root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/include/
                [...]
                drwxr-xr-x root/root         0 2019-12-16 21:26 TensorRT-7.0.0.11/data/
                drwxr-xr-x root/root         0 2019-12-16 21:20 TensorRT-7.0.0.11/data/mnist/
                [...]
                drwxr-xr-x root/root         0 2019-12-16 21:20 TensorRT-7.0.0.11/data/faster-rcnn/
                [...]
                drwxr-xr-x root/root         0 2019-12-16 21:20 TensorRT-7.0.0.11/data/resnet50/
                [...]
                drwxr-xr-x root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/doc/
                [...]
                lrwxrwxrwx root/root         0 2019-12-16 21:21 TensorRT-7.0.0.11/TensorRT-Release-Notes.pdf -> doc/pdf/TensorRT-Release-Notes.pdf
                drwxr-xr-x root/root         0 2019-12-16 21:19 TensorRT-7.0.0.11/python/
                -rw-r--r-- root/root    585782 2019-12-16 21:19 TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp27-none-linux_x86_64.whl
                -rw-r--r-- root/root    572613 2019-12-16 21:19 TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp34-none-linux_x86_64.whl
                -rw-r--r-- root/root    573944 2019-12-16 21:19 TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp35-none-linux_x86_64.whl
                -rw-r--r-- root/root    573910 2019-12-16 21:19 TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl
                -rw-r--r-- root/root    570976 2019-12-16 21:19 TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl
                drwxr-xr-x root/root         0 2019-12-16 21:19 TensorRT-7.0.0.11/uff/
                -rw-r--r-- root/root     60047 2019-12-16 21:19 TensorRT-7.0.0.11/uff/uff-0.6.5-py2.py3-none-any.whl
                drwxr-xr-x root/root         0 2019-12-16 21:19 TensorRT-7.0.0.11/graphsurgeon/
                -rw-r--r-- root/root     16930 2019-12-16 21:19 TensorRT-7.0.0.11/graphsurgeon/graphsurgeon-0.4.1-py2.py3-none-any.whl

        conda search tensorrt
            No match found for: tensorrt. Search: *tensorrt*
        cd /opt/packages/TensorFlow/gnu/TensorRT-7.0.0.11/
        # CWD: /opt/packages/TensorFlow/gnu/TensorRT-7.0.0.11/
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/packages/TensorFlow/gnu/TensorRT-7.0.0.11/lib"
        which pip
            /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/bin/pip
        cd python/
        # CWD: /opt/packages/TensorFlow/gnu/TensorRT-7.0.0.11/python/
        python --version
            Python 3.7.7
        pip install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl
            Processing ./tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl
            Installing collected packages: tensorrt
            Successfully installed tensorrt-7.0.0.11
        cd ../uff
        # CWD: /opt/packages/TensorFlow/gnu/TensorRT-7.0.0.11/uff/
        conda search uff
            No match found for: uff. Search: *uff*
        pip install uff-0.6.5-py2.py3-none-any.whl
            Processing ./uff-0.6.5-py2.py3-none-any.whl
            Requirement already satisfied: protobuf>=3.3.0 in /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages (from uff==0.6.5) (3.11.4)
            Requirement already satisfied: numpy>=1.11.0 in /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages (from uff==0.6.5) (1.18.1)
            Requirement already satisfied: six>=1.9 in /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages (from protobuf>=3.3.0->uff==0.6.5) (1.14.0)
            Requirement already satisfied: setuptools in /opt/packages/TensorFlow/gnu/tf1.8_py3_gpu_conda/lib/python3.7/site-packages (from protobuf>=3.3.0->uff==0.6.5) (46.1.1.post20200323)
            Installing collected packages: uff
            Successfully installed uff-0.6.5
        cd ../graphsurgeon/
        # CWD: /opt/packages/TensorFlow/gnu/TensorRT-7.0.0.11/graphsurgeon/
        conda search graphsurgeon
            No match found for: graphsurgeon. Search: *graphsurgeon*
        pip install graphsurgeon-0.4.1-py2.py3-none-any.whl
            Processing ./graphsurgeon-0.4.1-py2.py3-none-any.whl
            Installing collected packages: graphsurgeon
            Successfully installed graphsurgeon-0.4.1

    Test the install
        python
            Python 3.7.7 (default, Mar 26 2020, 15:48:22)
            [GCC 7.3.0] :: Anaconda, Inc. on linux
            Type "help", "copyright", "credits" or "license" for more information.
            >>> import tensorflow as tf
            >>> print(tf.__version__)
    OK      2.1.0
            >>>
            >>> print("\n")


            >>> print("### PSC: List GPUs available: %s" % tf.config.list_physical_devices('GPU'))
            2020-03-28 16:06:52.445732: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
            2020-03-28 16:06:52.468644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
            pciBusID: 0000:87:00.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
            coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
            2020-03-28 16:06:52.510030: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
            2020-03-28 16:06:52.645236: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
            2020-03-28 16:06:52.750363: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
            2020-03-28 16:06:52.910965: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
            2020-03-28 16:06:53.000383: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
            2020-03-28 16:06:53.101351: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
            2020-03-28 16:06:53.211366: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
            2020-03-28 16:06:53.214257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
    OK      ### PSC: List GPUs available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
            >>>
            >>> print("\n")


            >>> print("### PSC: Is built with CUDA?: %s" % tf.test.is_built_with_cuda())
    OK      ### PSC: Is built with CUDA?: True
            >>>
            >>> print("\n")


            >>> print("### PSC: Is executing eagerly?: %s" % tf.executing_eagerly())
            ### PSC: Is executing eagerly?: True
            >>>
            >>> print("\n")


            >>> print("### PSC: Running test code using /gpu:0")
            ### PSC: Running test code using /gpu:0
            >>> with tf.compat.v1.Session() as sess:
            ...     with tf.device('/gpu:0'):
            ...         a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            ...         b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            ...         c = tf.matmul(a, b)
            ...
            ...         print("\n")
            ...         print("### PSC: Running test on GPU. If a 2x2 matrix is shown, TF over GPU is working.")
            ...         print(sess.run(c))
            ...
            2020-03-28 16:06:58.081783: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
            2020-03-28 16:06:58.101415: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095054999 Hz
            2020-03-28 16:06:58.109678: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f99ba52b30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
            2020-03-28 16:06:58.109716: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
            2020-03-28 16:06:58.112376: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
            pciBusID: 0000:87:00.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
            coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
            2020-03-28 16:06:58.112439: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
            2020-03-28 16:06:58.112460: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
            2020-03-28 16:06:58.112479: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
            2020-03-28 16:06:58.112495: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
            2020-03-28 16:06:58.112512: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
            2020-03-28 16:06:58.112529: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
            2020-03-28 16:06:58.112554: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
            2020-03-28 16:06:58.115423: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
            2020-03-28 16:06:58.115468: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
            2020-03-28 16:06:58.244025: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
            2020-03-28 16:06:58.244064: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0
            2020-03-28 16:06:58.244077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N
            2020-03-28 16:06:58.248186: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15216 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:87:00.0, compute capability: 6.0)
            2020-03-28 16:06:58.250766: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f99c1d95f0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    OK      2020-03-28 16:06:58.250787: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla P100-PCIE-16GB, Compute Capability 6.0


            ### PSC: Running test on GPU. If a 2x2 matrix is shown, TF over GPU is working.
    OK      [[22. 28.]
            [49. 64.]]
            >>>
## Tests

### TF2 webpage, advanced example, using Keras. (tensorflow/tensorflow2/tests/advanced_tutorial.py)
https://www.tensorflow.org/tutorials/quickstart/advanced
    
    python tensorflow/tensorflow2/tests/advanced_tutorial.py
            2020-03-28 16:16:34.424771: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    OK      2020-03-28 16:16:34.457136: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
            pciBusID: 0000:87:00.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
            coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
            2020-03-28 16:16:34.457969: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
            2020-03-28 16:16:34.460980: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
            2020-03-28 16:16:34.463593: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
            2020-03-28 16:16:34.464359: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
            2020-03-28 16:16:34.467200: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
            2020-03-28 16:16:34.468645: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
            2020-03-28 16:16:34.474283: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
            2020-03-28 16:16:34.476934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
            2020-03-28 16:16:34.477465: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
            2020-03-28 16:16:34.492893: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095054999 Hz
            2020-03-28 16:16:34.498390: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5587f18bca90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
            2020-03-28 16:16:34.498421: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
            2020-03-28 16:16:34.500632: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
            pciBusID: 0000:87:00.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
            coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
            2020-03-28 16:16:34.500673: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
            2020-03-28 16:16:34.500689: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
            2020-03-28 16:16:34.500701: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
            2020-03-28 16:16:34.500714: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
            2020-03-28 16:16:34.500726: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
            2020-03-28 16:16:34.500738: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
            2020-03-28 16:16:34.500750: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    OK      2020-03-28 16:16:34.503446: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
            2020-03-28 16:16:34.503480: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
            2020-03-28 16:16:34.618757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
            2020-03-28 16:16:34.618797: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
            2020-03-28 16:16:34.618811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
            2020-03-28 16:16:34.622859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15216 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:87:00.0, compute capability: 6.0)
            2020-03-28 16:16:34.625397: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5587f1fe0340 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
            2020-03-28 16:16:34.625419: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla P100-PCIE-16GB, Compute Capability 6.0
            WARNING:tensorflow:Layer my_model is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
            If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
            To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
            2020-03-28 16:16:37.393657: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
            2020-03-28 16:16:37.744981: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
            Epoch 1, Loss: 0.1334172785282135, Accuracy: 95.97666931152344, Test Loss: 0.06454825401306152, Test Accuracy: 97.88999938964844
            Epoch 2, Loss: 0.04284348711371422, Accuracy: 98.69667053222656, Test Loss: 0.053079940378665924, Test Accuracy: 98.27999877929688
            Epoch 3, Loss: 0.021518589928746223, Accuracy: 99.2699966430664, Test Loss: 0.052586399018764496, Test Accuracy: 98.45999908447266
            Epoch 4, Loss: 0.013838252983987331, Accuracy: 99.5433349609375, Test Loss: 0.059811320155858994, Test Accuracy: 98.43000030517578
    OK      Epoch 5, Loss: 0.00949440523982048, Accuracy: 99.6500015258789, Test Loss: 0.06168389320373535, Test Accuracy: 98.3499984741211

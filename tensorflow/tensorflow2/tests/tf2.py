import tensorflow as tf

print(tf.__version__)

print("\n")
print("### PSC: List GPUs available: %s" % tf.config.list_physical_devices('GPU'))

print("\n")
print("### PSC: Is built with CUDA?: %s" % tf.test.is_built_with_cuda())

print("\n")
print("### PSC: Is executing eagerly?: %s" % tf.executing_eagerly())

print("\n")
print("### PSC: Running test code using /gpu:0")
with tf.compat.v1.Session() as sess:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

        print("\n")
        print("### PSC: Running test on GPU. If a 2x2 matrix is shown, TF over GPU is working.")
        print(sess.run(c))
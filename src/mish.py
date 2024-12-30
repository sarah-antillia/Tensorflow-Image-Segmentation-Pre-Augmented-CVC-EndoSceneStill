# mish.py

import tensorflow as tf

# Seel also: https://arxiv.org/abs/2107.12461
@tf.function
def mish(x):
    # 2024/09/01 commnet out the following line
    #x = tf.convert_to_tensor(x) #Added this line
    #print(">>> mish---")
    return tf.math.multiply(x, tf.math.tanh(tf.math.softplus(x)))

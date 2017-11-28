import tensorflow as tf
import numpy as np

labell_holder = tf.placeholder(dtype=tf.string, shape=[None,])
labelSplitter = tf.string_split(labell_holder, delimiter=",")
tensor = tf.sparse_tensor_to_dense(labelSplitter, default_value="")
tensor = tf.string_to_number(tensor)
tensor = tf.transpose(tensor, perm=[1,0])

x_ind = tf.cast(tensor[:, :1] * 7 / 448, dtype=tf.int64)
y_ind = tf.cast(tensor[:, 1:2] * 7 / 448, dtype=tf.int64)

labels = tf.zeros((7, 7, 25))





# xMinTensor = tensor[:, :1]
# yMinTensor = tensor[:, 1:2]
# xMaxTensor = tensor[:, 2:3]
# yMaxTensor = tensor[:, 3:4]
# labelTensor = tensor[:, 4:5]

# label = np.zeros((7, 7, 25))



# x_ind = tf.cast(tensor[:, :1] * 7 / 224, dtype=tf.int64)
# y_ind = tf.cast(tensor[:, 1:2] * 7 / 224, dtype=tf.int64)



# shape = tensor.get_shape()
# tensor =tensor[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xmin = "48,8"
    ymin = "240,12"
    xmax = "195,352"
    ymax = "371,498"
    label = "11,14"
    print(sess.run([labels, x_ind, y_ind],feed_dict={labell_holder : [xmin, ymin, xmax, ymax,label]}))
    print('----------')

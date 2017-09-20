import tensorflow as tf
from six.moves import xrange
import os
import time
import sys
'''
images:[batch_size, rows, cols, depth]
labels:[batch_size, label]
data_dir:tfrecords output path
filename:tfrecords file name
'''
def _convertToTFRecords(images, labels, batch_size, data_dir, filename):
    """
    Args:
        images: (num_examples, height, width, channels) np.int64 nparray (0~255)
        labels: (num_examples) np.int64 nparray
        num_examples: number of examples
        filename: the tfrecords' name to be saved
    Return: None, but store a .tfrecords file to data_log/
    """
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, filename))
    for index in xrange(batch_size):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())
        percent = (index / batch_size) * 100
        if (percent % 1 == 0):
            s1 = "\rTFRecords[%s%s]%d%%"%("*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1))
            sys.stdout.write(s1)
            sys.stdout.flush()
            time.sleep(0.3)
    writer.close()
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
def _convertToTFRecords(images, labels, batch_size, data_dir, filename, perfilerecordcount = 2500):
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
    perfileRecordNow = 0
    fileIndex = 0
    IsLabelsInt = True
    if len(labels.shape) > 1: # bytes
        IsLabelsInt = False

    for index in range(batch_size):
        image_raw = images[index].tostring()
        label_raw = labels[index].tostring() if IsLabelsInt == False else labels[index]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'label_raw' if IsLabelsInt == False else 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])) if IsLabelsInt == False else tf.train.Feature(int64_list=tf.train.Int64List(value=[label_raw])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))

        if perfileRecordNow == perfilerecordcount:
            print("\r%d file closed. %d" % (fileIndex, perfileRecordNow))
            writer.close()
            fileIndex += 1
            perfileRecordNow = 0

        if perfileRecordNow == 0:
            writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, "%s_%d.tfrecords" % (filename, fileIndex)))


        writer.write(example.SerializeToString())
        perfileRecordNow += 1
        percent = (index / batch_size) * 100
        if (percent % 1 == 0):
            s1 = "\rTFRecords[%s%s]%d%%"%("*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1))
            sys.stdout.write(s1)
            sys.stdout.flush()
            # time.sleep(0.3)
    if perfileRecordNow > 0:
        print("\r%d file closed. %d" % (fileIndex, perfileRecordNow))
        writer.close()
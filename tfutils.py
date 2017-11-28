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
class TFMaker():
    def __init__(self, fileList, batchSize, tfFileDir):
        self.FileList = fileList
        self.batchSize = batchSize
        self.tfFileDir = tfFileDir

    def Make(self):
        with tf.python_io.TFRecordWriter(self.tfFileDir) as writer:
            for index in range(self.batchSize):
                filename = tf.compat.as_bytes(os.path.basename(self.FileList[index]))

                example = tf.train.Example(features=tf.train.Features(feature={
                    'file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename]))
                }))

                writer.write(example.SerializeToString())

class TFReader():
    def __init__(self, filelist, batch_size, num_threads=2, min_after_dequeue=1000):
        self.FileList = filelist
        self.batchSize = batch_size
        self.numThreads = num_threads
        self.minAfterDequeue = min_after_dequeue

    def Read(self):
        filename_queue = tf.train.string_input_producer(self.FileList)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'file': tf.FixedLenFeature([], tf.string)
            }
        )
        fileList = tf.cast(features['file'], tf.string)
        # tf.train.shuffle_batch internally uses a RandomShuffleQueue
        file = tf.train.shuffle_batch(
            [fileList], batch_size=self.batchSize, num_threads=self.numThreads,
            min_after_dequeue=self.minAfterDequeue,
            capacity=self.minAfterDequeue + (self.numThreads + 1) * self.batchSize
        )

        return file
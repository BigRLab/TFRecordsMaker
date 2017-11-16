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
    def __init__(self, images, labels, batchSize, tfFileDir, tfFileNamePrefix, perFileRecordCount=2500):
        self.Images = images
        self.Labels = labels
        self.batchSize = batchSize
        self.tfFileDir = tfFileDir
        self.namePrefix = tfFileNamePrefix
        self.perFileCount = perFileRecordCount

    def Make(self, *data):
        perfileRecordNow = 0
        fileIndex = 0

        for index in range(self.batchSize):
            example = self._maker(index, *data)

            if perfileRecordNow == self.perFileCount:
                print("\r%d file closed. %d" % (fileIndex, perfileRecordNow))
                writer.close()
                fileIndex += 1
                perfileRecordNow = 0

            if perfileRecordNow == 0:
                writer = tf.python_io.TFRecordWriter(os.path.join(self.tfFileDir, "%s_%d.tfrecords" % (self.namePrefix, fileIndex)))

            writer.write(example.SerializeToString())
            perfileRecordNow += 1
            percent = (index / self.batchSize) * 100
            if (percent % 1 == 0):
                s1 = "\rTFRecords[%s%s]%d%%"%("*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1))
                sys.stdout.write(s1)
                sys.stdout.flush()
                # time.sleep(0.3)
        if perfileRecordNow > 0:
            print("\r%d file closed. %d" % (fileIndex, perfileRecordNow))
            writer.close()

    def _maker(self, index, *data):
        pass

class ClassficationMaker(TFMaker):
    def _maker(self, index, *data):
        rows = self.Images.shape[1]
        cols = self.Images.shape[2]
        depth = self.Images.shape[3]

        images = self.Images[index].tostring()
        classes = self.Labels[index]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'classes' : tf.train.Feature(int64_list=tf.train.Int64List(value=[classes])),
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images]))
        }))
        return example

class DetectionMaker(TFMaker):
    def _maker(self, index, *data):
        rows = self.Images.shape[1]
        cols = self.Images.shape[2]
        depth = self.Images.shape[3]
        filename = data[0][index]
        boxes = self.Labels[index]

        classes =  [str(b['classes']) for b in boxes]
        classes = tf.compat.as_bytes(os.path.basename(",".join(classes)))

        box = [b['box'] for b in boxes]
        boxes = ";".join(box)
        boxes = tf.compat.as_bytes(os.path.basename(boxes))

        base_name = tf.compat.as_bytes(os.path.basename(filename))
        images = self.Images[index].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'classes' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes])),
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images])),
            'filename' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[base_name])),
            'bBoxes':tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes]))
        }))
        return example

class TFReader():
    def __init__(self, filelist, batch_size, num_threads=2, min_after_dequeue=1000):
        self.FileList = filelist
        self.batchSize = batch_size
        self.numThreads = num_threads
        self.minAfterDequeue = min_after_dequeue

    def Read(self, *shape):
        filename_queue = tf.train.string_input_producer(self.FileList)

        list = self._read_And_decode(filename_queue, *shape) # share filename_queue with multiple threads

        # tf.train.shuffle_batch internally uses a RandomShuffleQueue
        list = tf.train.shuffle_batch(
            list, batch_size=self.batchSize, num_threads=self.numThreads,
            min_after_dequeue=self.minAfterDequeue,
            capacity=self.minAfterDequeue + (self.numThreads + 1) * self.batchSize
        )

        return list

    def _read(self, serialized_example, *shape):
        pass

    def _read_And_decode(self, filename_queue, *shape):
        """Return a single example for queue"""
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        return self._read(serialized_example, *shape)

class ClassficationReader(TFReader):
    def _read(self, serialized_example, *shape):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'images': tf.FixedLenFeature([], tf.string),
                'classes': tf.FixedLenFeature([], tf.int64)
            }
        )
        # some essential steps

        image = tf.decode_raw(features['images'], tf.uint8)

        imageShape = shape[0]
        image = tf.reshape(image, imageShape)    # THIS IS IMPORTANT
        image.set_shape(imageShape)
        image = tf.cast(image, tf.float32) * (1 / 255.0)  # set to [0, 1]

        sparse_label = tf.cast(features['classes'], tf.int64)
        # sparse_label = tf.reshape(sparse_label, shape[1])
        #
        return [image, sparse_label]

class DetectionReader(TFReader):
    def _read(self, serialized_example, *shape):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'images': tf.FixedLenFeature([], tf.string),
                'filename': tf.FixedLenFeature([], tf.string),
                'classes': tf.FixedLenFeature([], tf.string),
                'bBoxes':tf.FixedLenFeature([], tf.string)
            }
        )
        # some essential steps

        image = tf.decode_raw(features['images'], tf.uint8)

        imageShape = shape[0]
        image = tf.reshape(image, imageShape)    # THIS IS IMPORTANT
        image.set_shape(imageShape)
        image = tf.cast(image, tf.float32) * (1 / 255.0)  # set to [0, 1]

        sparse_label = tf.cast(features['classes'], tf.string)

        filename = tf.cast(features['filename'], tf.string)
        bBoxes = tf.cast(features['bBoxes'], tf.string)
        return [filename,image,sparse_label, bBoxes ]
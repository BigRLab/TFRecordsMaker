import tfRecordsMaker
import tflearn
from PIL import Image
import os
import numpy as np
import time
import sys
import tfRecordsReader
import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_SIZE = 224
IMAGE_CHANNEL = 3
DATA_DIR = 'test'

# train :25000
# test : 1300
def ConverToImage(img_path):
    img = Image.open(img_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    img.load()
    img = img.convert("RGB")
    array = np.asarray(img, dtype=np.uint8)
    # array /= 255.

    return array.reshape([IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
def getAllJPGFileName():
    g = os.walk(DATA_DIR)
    fileList = []
    for path,d,filelist in g:
        for filename in filelist:
            if filename.endswith('jpg'):
                if "dog" in filename:
                    fileList.append([os.path.join(path, filename), 1])
                else:
                    fileList.append([os.path.join(path, filename), 0])
    return fileList

if __name__ == '__main__':
    # plt.imshow(ConverToImage('ccc/cat.0.jpg'))
    # plt.show()
    fileList = getAllJPGFileName()
    batch_size = len(fileList)
    print(batch_size)
    image_list = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype=np.uint8)
    label_list = np.zeros((batch_size), dtype=np.int32)
    print(batch_size)
    for i,file in enumerate(fileList):
        filename = file[0]
        label = file[1]
        image = ConverToImage(filename)

        image_list[i] = image
        label_list[i] = label
        percent = (i / batch_size) * 100
        if (percent % 1 == 0):
            s1 = "\rImage Loading[%s%s]%d%%"%("*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1))
            sys.stdout.write(s1)
            sys.stdout.flush()
            time.sleep(0.3)
        # print(percent)

    tfRecordsMaker._convertToTFRecords(images=image_list, labels=label_list, batch_size=batch_size,
                                       data_dir='', filename=DATA_DIR + '.tfrecords')
    print("\rDone")

    # image, label = \
    #     tfRecordsReader.readFromTFRecords('test.tfrecords', batch_size=10, img_shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    #
    # with tf.Session() as sess:
    #     tf.train.start_queue_runners()
    #     image_batch, label_batch = sess.run([image, label])
    #     # print(image_batch)
    #     print(label_batch)
    #     plt.imshow(image_batch[4])
    #     plt.show()
